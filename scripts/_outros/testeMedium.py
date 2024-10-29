# https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11#ff89

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup)
from scripts._outros._Utils import Utils

patch = f'../resultados/testeMedium-{datetime.now().strftime("%Y-%m-%d_%H-%M")}/'

def preprocess_dataset_balance(path, numberRegisters):
    df_dataset = pd.read_csv(path)
    df_dataset = df_dataset[['content', 'fake_review']]
    dfReal, dfFake = df_dataset[df_dataset.fake_review == False], df_dataset[df_dataset.fake_review == True]
    df_dataset = pd.concat([dfReal.sample(n=int(numberRegisters/2), random_state=42), 
                            dfFake.sample(n=int(numberRegisters/2), random_state=42)])

    # df_dataset['content'] = df_dataset['content'].apply(lambda x: x.replace('<br />', ''))
    # df_dataset['content'] = df_dataset['content'].replace(r'\s+', ' ', regex=True)
    df_dataset['fake_review'] = df_dataset['fake_review'].astype(int)

    return df_dataset

class FineTuningPipeline:

    def __init__(self,dataset,tokenizer,model,optimizer,epochs,max_length,batch_size,
            loss_function = nn.CrossEntropyLoss(), val_df_train_size = 0.15, test_size=0.15, seed = 42):

        self.df_dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.max_length = max_length
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.val_size = val_df_train_size
        self.test_size = test_size
        self.seed = seed

        if torch.cuda.is_available(): self.device = torch.device('cuda:0')
        else: self.device = torch.device('cpu')

        self.model.to(self.device)
        self.set_seeds()
        self.token_ids, self.attention_masks = self.tokenize_dataset()
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.create_dataloaders()
        self.scheduler = self.create_scheduler()
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.fine_tune()

    def tokenize(self, text):
        batch_encoder = self.tokenizer.encode_plus(
            text,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt')

        token_ids = batch_encoder['input_ids']
        attention_mask = batch_encoder['attention_mask']

        return token_ids, attention_mask

    def tokenize_dataset(self):
        token_ids, attention_masks = [], []

        for review in self.df_dataset['content']:
            tokens, masks = self.tokenize(review)
            token_ids.append(tokens)
            attention_masks.append(masks)

        return torch.cat(token_ids, dim=0), torch.cat(attention_masks, dim=0)

    def create_dataloaders(self):
        labels = torch.tensor(self.df_dataset['fake_review'].values)
        
        # 85% para treino e validação, 15% para teste
        train_val_ids, test_ids, train_val_masks, test_masks, train_val_labels, test_labels = train_test_split(
            self.token_ids, self.attention_masks, labels, test_size=self.test_size, random_state=self.seed)

        # 60% para treino e 20% para validação
        train_ids, val_ids, train_masks, val_masks, train_labels, val_labels = train_test_split(
            train_val_ids, train_val_masks, train_val_labels, test_size=self.val_size, random_state=self.seed)
        
        train_data = TensorDataset(train_ids, train_masks, train_labels)
        train_data = TensorDataset(train_val_ids, train_val_masks, train_val_labels)
        val_data = TensorDataset(val_ids, val_masks, val_labels)
        test_data = TensorDataset(test_ids, test_masks, test_labels)
        
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size)

        return train_dataloader, val_dataloader, test_dataloader
        return train_dataloader, test_dataloader

    def create_scheduler(self):
        num_training_steps = self.epochs * len(self.train_dataloader)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps)

        return scheduler

    def set_seeds(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def predict(self, operacao, data):
        self.model.eval()
        val_loss, val_f1, val_precision, val_recall = 0,0,0,0
        t0_val = datetime.now()
        all_logits, all_label_ids = [], []

        print(f'\n{operacao}:\n--------- {t0_val}')
        
        if operacao == 'Teste': dataloader = self.test_dataloader
        else: dataloader = self.val_dataloader
        
        for batch in dataloader:
            batch_token_ids = batch[0].to(self.device)
            batch_attention_mask = batch[1].to(self.device)
            batch_labels = batch[2].to(self.device)

            with torch.no_grad():
                (loss, logits) = self.model(
                    batch_token_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                    token_type_ids=None,
                    return_dict=False
                )

            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            all_logits.append(logits)
            all_label_ids.append(label_ids)

            val_loss += loss.item()

        all_logits = np.concatenate(all_logits, axis=0)
        all_label_ids = np.concatenate(all_label_ids, axis=0)

        predictions = np.argmax(all_logits, axis=1)

        val_f1 = f1_score(all_label_ids, predictions, average='weighted')
        val_precision = precision_score(all_label_ids, predictions, average='weighted')
        val_recall = recall_score(all_label_ids, predictions, average='weighted')

        print(classification_report(all_label_ids, predictions))

        average_val_loss = val_loss / len(self.val_dataloader)
        time_val = datetime.now() - t0_val

        print(f"Average Validation Loss: {average_val_loss:.4f} (temporizador {time_val})")
        print(f"{operacao}: F1-Score {val_f1:.4f}, Precision {val_precision:.4f}, Recall {val_recall:.4f}")

        return pd.DataFrame([{
            'epoch': self.epochs,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'learning_rate': LEARNING_RATE,
            'data': data,
            'operacao': operacao,
            'f1_score': round(val_f1, 4),
            'precision': round(val_precision, 4),
            'recall': round(val_recall, 4)
        }]).round(5)

    def fine_tune(self):
        t0_train = datetime.now()

        for epoch in range(0, self.epochs):
            self.model.train()
            training_loss = 0
            t0_epoch = datetime.now()
            
            print(f'{"-"*20} Epoch {epoch+1} {"-"*20}')
            print(f'Training:\n--------- {t0_epoch}')

            for step, batch in enumerate(self.train_dataloader):

                batch_token_ids = batch[0].to(self.device)
                batch_attention_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                self.model.zero_grad()

                loss, logits = self.model(
                    batch_token_ids,
                    token_type_ids = None,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                    return_dict=False)

                training_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                if step == 1 or step % 50 == 0 or step == 297:
                    print(f'Step {step + 1}/{len(self.train_dataloader)} - Learning Rate: {current_lr:.8f}')
                    Utils.SalvarCsv(patch, 'relatorio.csv', pd.DataFrame([{
                                                                'epoch': epoch,
                                                                'batch_size': self.batch_size,
                                                                'max_length': self.max_length,
                                                                'learning_rate': current_lr,
                                                                'data': t0_train,
                                                                'operacao': 'treino',
                                                                'f1_score': "-",
                                                                'precision': "-",
                                                                'recall': "-"
                                                            }]).round(5))
        
            average_train_loss = training_loss / len(self.train_dataloader)
            time_epoch = datetime.now() - t0_epoch

            print(f'Average Loss:     {average_train_loss}')
            print(f'Time Taken:       {time_epoch}')
            
            df_metrics = self.predict('Validação', t0_train)
            Utils.SalvarCsv(patch, 'relatorio.csv', df_metrics)
            
            if df_metrics.iloc[0]['f1_score'] > self.best_val_f1:
                self.best_val_f1 = df_metrics.iloc[0]['f1_score']
                self.best_model_state = self.model.state_dict()
                print(f"Novo melhor modelo encontrado com F1-Score: {self.best_val_f1:.4f}")
            
        print(f'Total training time: {datetime.now()-t0_train}')
        
        print(f'{"-"*20} Test {"-"*20}')
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Melhor modelo restaurado com F1-Score de validação: {self.best_val_f1:.4f}")
        
        df_metrics = self.predict('Teste', t0_train)
        Utils.SalvarCsv(patch, 'relatorio.csv', df_metrics)
    
    
LEARNING_RATE = 5e-5
URL_DATASET = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
DATASET = preprocess_dataset_balance(URL_DATASET, int(3387*2)) # 3387
TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
MODEL = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased',num_labels=2)
OPTIMIZER = AdamW(MODEL.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

fine_tuned_model = FineTuningPipeline(
    dataset = DATASET,
    tokenizer = TOKENIZER,
    model = MODEL,
    optimizer = OPTIMIZER,
    epochs = 30,
    batch_size = 16,
    max_length = 256,
)