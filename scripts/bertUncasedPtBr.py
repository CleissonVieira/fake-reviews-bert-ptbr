import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.utils import resample

path = f'./bert/{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
os.makedirs(path, exist_ok=True)

deviceGpu = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Usando o dispositivo: {deviceGpu}')

EPOCHS = 50 # Quantidade de Iteração completa sobre todos os dados de treinamento
BATCH_SIZE = 16 # Número de amostras processadas simultaneamente em cada etapa de treinamento
MAX_LEN = 128 # Comprimento máximo sequências tokens, número palavras/tokenizações que BERT processa por amostra
LEARNING_RATE=2e-5 # TESTAR COM VALIDAÇÃO CRUZADA: 2e-5, lr=3e-5 e lr=5e-6

# url = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-en.csv'
url = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
df = pd.read_csv(url)
df = df[['content', 'fake_review']]

# Balanceamento classe fake_review (True/False 30 registros de cada)
def BalancearComMenorQuantidade():
    num_real = len(df[df.fake_review == False])
    num_fakes = len(df[df.fake_review == True])
    df_real = df[df.fake_review == False].sample(n=min(num_fakes, num_real), random_state=42)
    df_fakes = df[df.fake_review == True].sample(n=min(num_fakes, num_real), random_state=42)
    return pd.concat([df_real, df_fakes])

def BalancerDuplicandoRegistroParaMenorQuantidade():
    df_real = df[df.fake_review == False]
    df_fakes = df[df.fake_review == True]
    if len(df_real) > len(df_fakes):
        df_fakes_upsampled = resample(df_fakes, replace=True, n_samples=len(df_real), random_state=123)
        return pd.concat([df_real, df_fakes_upsampled])
    else:
        df_real_upsampled = resample(df_real, replace=True, n_samples=len(df_fakes), random_state=123)
        return pd.concat([df_real_upsampled, df_fakes])
    
def DataframeTesteAmostraManual():
    df_real = df[df.fake_review == False].sample(n=1000, random_state=42)
    df_fakes = df[df.fake_review == True].sample(n=1000, random_state=42)
    return pd.concat([df_real, df_fakes])

df_balanceado = BalancearComMenorQuantidade()
df_balanceado['fake_review'] = df_balanceado['fake_review'].astype(int)

# Separando o dataset em treino (80%) e teste (20%)
train_df, test_df = train_test_split(df_balanceado, test_size=0.2, random_state=42, stratify=df_balanceado['fake_review'])

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        content = str(self.df.iloc[index, 0])
        label = self.df.iloc[index, 1]
        
        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'review_text': content,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Criando Dataloaders
def CriarDataLoader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(df, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

train_data_loader = CriarDataLoader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = CriarDataLoader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Carregando o modelo pré-treinado
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
model = model.to(deviceGpu)

# Configurando o otimizador, função de perda e scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
loss_fn = nn.CrossEntropyLoss().to(deviceGpu)
total_steps = len(train_data_loader) * 20

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def TreinarModeloEpoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def ValidarModelo(model, data_loader, loss_fn, device):
    model = model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with open(f'{path}/result_train.txt', 'w') as f:
    # Treinamento e validação
    for epoch in range(EPOCHS):
        train_loss = TreinarModeloEpoch(model, train_data_loader, loss_fn, optimizer, deviceGpu, scheduler)
        val_acc, val_loss = ValidarModelo(model, test_data_loader, loss_fn, deviceGpu)
        
        print(f'Epoch {epoch + 1}/{EPOCHS}:     Perda no treinamento {train_loss} | Acurácia na Validação {val_acc} | Perda na Validação {val_loss}')
        f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ')
        f.write(f'Epoch {epoch + 1}/{EPOCHS}:   Perda no treinamento: {train_loss:.4f} | Acurácia na Validação: {val_acc:.4f} | Perda na Validação: {val_loss:.4f}\n')

# Salvando o modelo treinado
model.save_pretrained(f'{path}/model')
tokenizer.save_pretrained(f'{path}/tokenizer')

# Função para realizar a predição no dataset de teste
def predict_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds)
            real_values.extend(labels)

    return torch.stack(predictions).cpu(), torch.stack(real_values).cpu()

# Realizando as predições no dataset de teste
y_pred, y_true = predict_model(model, test_data_loader, deviceGpu)

# Calculando as métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Exibindo as métricas
print(f'\n\nAcurácia: {accuracy:.4f}')
print(f'Precisão: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}\n\n')

with open(f'{path}/result_metrics.txt', 'w') as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    f.write(f'\n\nAcurácia: {accuracy:.4f}\n')
    f.write(f'Precisão: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1-Score: {f1:.4f}\n')

# Criando um dataframe para análise das predições
df_results = pd.DataFrame({'Real': y_true, 'Predito': y_pred})
df_results.to_csv(f'{path}/result_predicts.csv', index=False)
print(df_results.head())

# trocar train test split (80:20) para crossValidate (estudar como faz com BERT)
# Validar com Guilherme
# os resultados são mais confiáveis
# Para comparação de resultado justo entre os estudos seria necessário fazer

# Gerar outras opções para gráfico
# 'f1_score_mean': np.mean(cv_results['test_f1_score']), #media
# 'f1_score_variance': np.var(cv_results['test_f1_score'], ddof=1),
# 'f1_score_min': np.min(cv_results['test_f1_score']),
# 'f1_score_max': np.max(cv_results['test_f1_score']),

# Manter random state sempre 42

# Dataset final dos resultados Percisi
# https://github.com/lucaspercisi/yelp-fake-reviews-ptbr/blob/main/Results/global_metric_df.csv
# essa tabela tem todas as possibilidades de treinamento
# utilizado para criar gráficos (colab Percisi)
# utilizar esses dados para comparar os meus resultados
