# https://colab.research.google.com/drive/1B_ERSgQDLNOL8NPCvkn7s8h_vEdZ_szI?usp=sharing#scrollTo=hdgWD4vzL4EL

import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast, AdamW
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from scripts._outros._Utils import Utils

_patch = f'../resultadoScripts/sentenceClassification/'

urlPtBr = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
dfPtBr = pd.read_csv(urlPtBr)
dfPtBr = dfPtBr[['content', 'fake_review']]
dfReal, dfFake = dfPtBr[dfPtBr.fake_review == False], dfPtBr[dfPtBr.fake_review == True]

_epochs = 5
_batchSize = 16 
_learningRate = 4e-5
_maxLength = 256
_model = 'bert-base-multilingual-uncased'

df = pd.concat([dfReal.sample(n=100, random_state=42), 
                dfFake.sample(n=100, random_state=42)])

df['fake_review'] = df['fake_review'].astype(int)
# print(df.head())

print("\nDataset Statistics:")
# print(df['fake_review'].value_counts())

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.dataset = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        review_text = self.dataset.iloc[idx, 0]  # Assuming reviewText is the first column
        labels = self.dataset.iloc[idx, 1]  # Convert sentiment to numerical label

        # Tokenize the review text
        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,  # Add [CLS] token at the start for classification
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
          'review_text': review_text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(), # this is NOT self-attention!
          'labels': torch.tensor(labels, dtype=torch.long)
        }
        
tokenizer = BertTokenizerFast.from_pretrained(_model)

train_df, val_test_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['fake_review'])
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42, stratify=val_test_df['fake_review'])

train_dataset = ReviewDataset(train_df, tokenizer, _maxLength)
val_dataset = ReviewDataset(val_df, tokenizer, _maxLength)
test_dataset = ReviewDataset(test_df, tokenizer, _maxLength)

train_loader = DataLoader(train_dataset, batch_size=_batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=_batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=_batchSize, shuffle=False)

len(train_loader), len(test_loader)

class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomBertForSequenceClassification, self).__init__()
        self.distilbert = BertModel.from_pretrained(_model)
        self.pre_classifier = nn.Linear(768, 768)  # BERT's hidden size is 768
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]  # (batch_size, sequence_length, hidden_size)
        pooled_output = hidden_state[:, 0]  # we take the representation of the [CLS] token (first token)
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output) # regularization
        logits = self.classifier(pooled_output)
        return logits

model = CustomBertForSequenceClassification()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=_learningRate)
data = datetime.now()

for epoch in range(_epochs):
    model.train()
    running_loss = 0.0
    
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step()
        
        running_loss += loss.item()
        if i + 1 == 1 or (i + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}, running_loss: {running_loss}")
    
    # Validation after each epoch
    model.eval()
    val_correct, val_total = 0, 0
    val_preds, val_labels = [], []
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            val_preds.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds)
    val_recall = recall_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds)

    print(f"Validação epoch {epoch + 1}: F1-score {val_f1:.4f} | Recall {val_recall:.4f} | Precision {val_precision:.4f}")
    
    df_metrics =pd.DataFrame([{
            'epoch': _epochs,
            'batch_size': _batchSize,
            'max_length': _maxLength,
            'learning_rate': _learningRate,
            'data': data,
            'operacao': 'validacao',
            'f1_score': round(val_f1, 4),
            'precision': round(val_precision, 4),
            'recall': round(val_recall, 4)
    }]).round(5)
    Utils.SalvarCsv(_patch, 'relatorio.csv', df_metrics)

model.eval()
test_preds, test_labels = [], []
with torch.inference_mode():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=1)
        test_preds.extend(predictions.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_f1 = f1_score(test_labels, test_preds)
test_recall = recall_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds)

print(f"Teste: F1-score {val_f1:.4f} | Recall {val_recall:.4f} | Precision {val_precision:.4f}")

df_metrics =pd.DataFrame([{
        'epoch': _epochs,
        'batch_size': _batchSize,
        'max_length': _maxLength,
        'learning_rate': _learningRate,
        'data': data,
        'operacao': 'teste',
        'f1_score': round(val_f1, 4),
        'precision': round(val_precision, 4),
        'recall': round(val_recall, 4)
}]).round(5)
Utils.SalvarCsv(_patch, 'relatorio.csv', df_metrics)

for param in model.distilbert.parameters():
    param.requires_grad = False

torch.save(model.state_dict(), f'{_patch}model.pth')
