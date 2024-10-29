# https://colab.research.google.com/drive/1B_ERSgQDLNOL8NPCvkn7s8h_vEdZ_szI?usp=sharing#scrollTo=hdgWD4vzL4EL
# Parte final 

import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, AdamW
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from scripts._outros._Utils import Utils

_patch = f'../resultadoScripts/sentenceClassificationPre/'

urlPtBr = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
dfPtBr = pd.read_csv(urlPtBr)
dfPtBr = dfPtBr[['content', 'fake_review']]
dfReal, dfFake = dfPtBr[dfPtBr.fake_review == False], dfPtBr[dfPtBr.fake_review == True]

_epochs = 5
_batchSize = 16 
_learningRate = 5e-5
_maxLength = 256
_model = 'bert-base-multilingual-uncased'

df = pd.concat([dfReal.sample(n=100, random_state=42), 
                dfFake.sample(n=100, random_state=42)])

df['fake_review'] = df['fake_review'].astype(int)

class FakeReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe['content'].tolist()
        self.labels = dataframe['fake_review'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # Tokenização do texto
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizerFast.from_pretrained(_model)

train_df, val_test_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['fake_review'])
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42, stratify=val_test_df['fake_review'])

train_dataset = FakeReviewDataset(train_df, tokenizer, _maxLength)
val_dataset = FakeReviewDataset(val_df, tokenizer, _maxLength)
test_dataset = FakeReviewDataset(test_df, tokenizer, _maxLength)

train_dataloader = DataLoader(train_dataset, batch_size=_batchSize, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=_batchSize, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=_batchSize, shuffle=False)

model = BertForSequenceClassification.from_pretrained(_model, num_labels=2)
optimizer = AdamW(model.parameters(), lr=_learningRate)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    
    df_metrics =pd.DataFrame([{
        'epoch': _epochs,
        'batch_size': _batchSize,
        'max_length': _maxLength,
        'learning_rate': _learningRate,
        'data': data,
        'operacao': 'validação',
        'f1_score': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4)
    }]).round(5)
    Utils.SalvarCsv(_patch, 'relatorio.csv', df_metrics)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

training_args = TrainingArguments(
    output_dir=_patch,          
    num_train_epochs=_epochs,              
    per_device_train_batch_size=_batchSize,   
    per_device_eval_batch_size=_batchSize,
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir=f'{_patch}/logs',           
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,          
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

data = datetime.now()
trainer.train()
model.save_pretrained(f'{_patch}model')
tokenizer.save_pretrained(f'{_patch}model')

test_results = trainer.evaluate(test_dataset) 
df_metrics =pd.DataFrame([{
    'epoch': _epochs,
    'batch_size': _batchSize,
    'max_length': _maxLength,
    'learning_rate': _learningRate,
    'data': data,
    'operacao': 'teste',
    'f1_score': round(test_results['eval_f1'], 4),
    'precision': round(test_results['eval_precision'], 4),
    'recall': round(test_results['eval_recall'], 4)
}]).round(5)
Utils.SalvarCsv(_patch, 'relatorio.csv', df_metrics)
print("Teste: ", test_results)
