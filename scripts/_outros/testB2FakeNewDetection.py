# https://colab.research.google.com/drive/1KPShsnIudbmI_y-brdDEcdPYLQSbtZqQ#scrollTo=HIFgZcfWlC8Fimport numpy as np

from datetime import datetime
import numpy as np
import pandas as pd
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from scripts._outros._Utils import Utils

torch.serialization.add_safe_globals
device = torch.device("cuda")

patch = f'../resultadoScripts/testX1FakeNewDetection-{datetime.now().strftime("%Y-%m-%d_%H-%M")}/'

urlPtBr = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
dfPtBr = pd.read_csv(urlPtBr)
dfPtBr = dfPtBr[['content', 'fake_review']]
dfReal, dfFake = dfPtBr[dfPtBr.fake_review == False], dfPtBr[dfPtBr.fake_review == True]

_epochs = 20
_batchSize = 16 
_learningRate = 3e-5
_maxLength = 256

# Load Dataset
true_data = dfReal.sample(n=3387, random_state=42)
fake_data = dfFake.sample(n=3387, random_state=42)

# Merge 'true_data' and 'fake_data', by random mixing into a single df called 'data'
data = pd.concat([true_data, fake_data]).reset_index(drop=True)


# See how the data looks like
print(data.shape)
data.head()

data['label'] = data['fake_review'].astype(int)
data.head()

# Checking if our data is well balanced
label_size = [data['label'].sum(),len(data['label'])-data['label'].sum()]
plt.pie(label_size,explode=[0.1,0.1],colors=['firebrick','navy'],startangle=90,shadow=True,labels=['True','False'],autopct='%1.1f%%')



# Train-Validation-Test set split into 70:15:15 ratio
# Train-Temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(data['content'], data['label'],
                                                                    random_state=42,
                                                                    test_size=0.3,
                                                                    stratify=data['fake_review'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=42,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# Load BERT model and tokenizer via HuggingFace Transformers
bert = AutoModel.from_pretrained('bert-base-multilingual-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')

# Plot histogram of the number of words in train data 'title'
seq_len = [len(content.split()) for content in train_text]

pd.Series(seq_len).hist(bins = 40,color='firebrick')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')

# Tokenize and encode sequences in the train set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = _maxLength,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = _maxLength,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = _maxLength,
    pad_to_max_length=True,
    truncation=True
)

# Convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids']).to(device)
test_mask = torch.tensor(tokens_test['attention_mask']).to(device)
test_y = torch.tensor(test_labels.tolist())

# Data Loader structure definition
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
_batchSize = 32                                               #define a batch size

train_data = TensorDataset(train_seq, train_mask, train_y)    # wrap tensors
train_sampler = RandomSampler(train_data)                     # sampler for sampling the data during training
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=_batchSize)
                                                              # dataLoader for train set
val_data = TensorDataset(val_seq, val_mask, val_y)            # wrap tensors
val_sampler = SequentialSampler(val_data)                     # sampler for sampling the data during training
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=_batchSize)


test_data = TensorDataset(test_seq, test_mask, test_y)            # wrap tensors
test_sampler = SequentialSampler(test_data)                     # sampler for sampling the data during training
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=_batchSize)
                                                 
                                                              # Freezing the parameters and defining trainable BERT structure
for param in bert.parameters():
    param.requires_grad = False    # false here means gradient need not be computed
    
class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x

model = BERT_Arch(bert)
model = model.to(device)
# Defining the hyperparameters (optimizer, weights of the classes and the epochs)
# Define the optimizer
from transformers import AdamW
optimizer = AdamW(model.parameters(),
                  lr = _learningRate)          # learning rate
# Define the loss function
cross_entropy  = nn.NLLLoss()

# Defining training and evaluation functions
def train():
  model.train()
  total_loss = 0

  for step,batch in enumerate(train_dataloader):                # iterate over batches
    if step % 50 == 0 and not step == 0:                        # progress update after every 50 batches.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
    
    batch = [r.to(device) for r in batch]                                  # push the batch to gpu
    sent_id, mask, labels = batch
    model.zero_grad()                                           # clear previously calculated gradients
    preds = model(sent_id, mask)                                # get model predictions for current batch
    loss = cross_entropy(preds, labels)                         # compute loss between actual & predicted values
    
    total_loss = total_loss + loss.item()                       # add on to the total loss
    loss.backward()                                             # backward pass to calculate the gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # clip gradients to 1.0. It helps in preventing exploding gradient problem
    optimizer.step()                                            # update parameters
    preds=preds.detach().cpu().numpy()                          # model predictions are stored on GPU. So, push it to CPU

  avg_loss = total_loss / len(train_dataloader)                 # compute training loss of the epoch
                                                                # reshape predictions in form of (# samples, # classes)
  return avg_loss                                 # returns the loss and predictions

def evaluate(dataloader):
  print("\nEvaluating...")
  model.eval()                                    # Deactivate dropout layers
  total_loss = 0
  
  all_preds,all_labels = [],[]
  
  for step,batch in enumerate(dataloader):    # Iterate over batches
    if step % 50 == 0 and not step == 0:          # Progress update every 50 batches.
                                                  # Calculate elapsed time in minutes.
                                                  # Elapsed = format_time(time.time() - t0)
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
                                                  # Report progress
    sent_id, mask, labels = [item.to(device) for item in batch]
    all_labels.extend(labels.detach().cpu().numpy())
            
    with torch.no_grad():                         # Deactivate autograd
      preds = model(sent_id, mask)                # Model predictions
      loss = cross_entropy(preds,labels)          # Compute the validation loss between actual and predicted values
      total_loss = total_loss + loss.item()
      
      preds = preds.detach().cpu().numpy()
      all_preds.extend(np.argmax(preds, axis=1))
      
  avg_loss = total_loss / len(dataloader)         # compute the validation loss of the epoch
  precision = precision_score(all_labels, all_preds, average='weighted')
  recall = recall_score(all_labels, all_preds, average='weighted')
  f1 = f1_score(all_labels, all_preds, average='weighted')
  
  return avg_loss, precision, recall, f1

# Train and predict
best_valid_loss = float('inf')
train_losses,valid_losses=[],[]
precision, recall, f1 = 0,0,0

data = datetime.now()
for epoch in range(_epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, _epochs))
    train_loss = train()                       # train model
    valid_loss, precision, recall, f1 = evaluate(val_dataloader)                    # evaluate model
    print(f'Treinamento: F1-Score: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}')
    
    df_metrics = pd.DataFrame([{
            'epoch': _epochs,
            'batch_size': _batchSize,
            'max_length': _maxLength,
            'learning_rate': _learningRate,
            'data': data,
            'operacao': 'train',
            'f1_score': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4)
    }]).round(5)
    Utils.SalvarCsv(patch, 'relatorio.csv', df_metrics)
    
    if valid_loss < best_valid_loss:              # save the best model
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'{patch}model.pth')
        print(f'model save metrics: ', df_metrics)
    train_losses.append(train_loss)               # append training and validation loss
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
    
# load weights of best model
# path = 'c1_fakenews_weights.pt'
model.load_state_dict(torch.load(f'{patch}model.pt', weights_only=True))
model.to(device)

with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))

precision = precision_score(test_y, preds, average='weighted')
recall = recall_score(test_y, preds, average='weighted')
f1 = f1_score(test_y, preds, average='weighted')
print(f'Teste 1: F1-Score {f1:.3f} | Precision {precision:.3f} | Recall {recall:.3f}')

df_metrics =pd.DataFrame([{
            'epoch': _epochs,
            'batch_size': _batchSize,
            'max_length': _maxLength,
            'learning_rate': _learningRate,
            'data': data,
            'operacao': 'test_1',
            'f1_score': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4)
}]).round(5)
Utils.SalvarCsv(patch, 'relatorio.csv', df_metrics)

valid_loss, precision, recall, f1 = evaluate(test_dataloader) 
print(f'Teste 2: F1-Score {f1:.3f} | Precision {precision:.3f} | Recall {recall:.3f}')

df_metrics = pd.DataFrame([{
            'epoch': _epochs,
            'batch_size': _batchSize,
            'max_length': _maxLength,
            'learning_rate': _learningRate,
            'data': data,
            'operacao': 'test_2',
            'f1_score': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4)
}]).round(5)
Utils.SalvarCsv(patch, 'relatorio.csv', df_metrics)