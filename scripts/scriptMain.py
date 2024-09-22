# REQUISITOS PARA EXECUÇÃO
# pip install pandas
# pip install scikit-learn
# pip install datasets
# pip install datasets
# pip install transformers[torch]
# pip install accelerate -U

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def MetricasPersonalizadas(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    print(f'\n\nMÉTRICAS')
    print(f'Predictions: {predictions}')
    print(f'Labels: {labels}\n')
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1-score': f1,
        'precision': precision,
        'recall': recall
    }
    
def ObterDfBalanceado(df_verdadeiros, df_falsos):
    num_falsos = df_falsos.shape[0] # Quantidade de registros em cada classe
    num_verdadeiros = df_verdadeiros.shape[0]

    if num_falsos > num_verdadeiros: # Amostra aleatória da classe com mais registros
        df_falsos = df_falsos.sample(num_verdadeiros, random_state=42)  # Usando estado aleatório para reprodutibilidade
    else: df_verdadeiros = df_verdadeiros.sample(num_falsos, random_state=42)

    df_balanceado = pd.concat([df_falsos, df_verdadeiros])
    df_balanceado = df_balanceado[['content', 'fake_review']] # Filtra apenas coluna 'content' (texto das avaliações) e 'fake_review' (rótulos)
    df_balanceado['fake_review'] = df_balanceado['fake_review'].astype(int)  # Converte coluna 'fake_review' True/False para 1/0
    
    return df_balanceado


# PREPARAÇÃO E DIVISÃO DOS DADOS
# url = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-en.csv'
url = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
df_original = pd.read_csv(url)

df_falsos = df_original[df_original['fake_review'] == False] # Separando classe verdadeiro e falso
df_verdadeiros = df_original[df_original['fake_review'] == True]

df_sample = ObterDfBalanceado(df_verdadeiros, df_falsos)

train_texts, test_texts, train_labels, test_labels = train_test_split( # Divide dataset em treino e teste (80/20)
    df_sample['content'], df_sample['fake_review'], test_size=0.3, random_state=42
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # Carregar o tokenizer DistilBERT
def tokenize_function(examples): # Tokeniza as entradas
    return tokenizer(examples['content'], truncation=True, padding=True, max_length=256)

train_dataset = Dataset.from_dict({'content': train_texts, 'labels': train_labels}) # Cria datasets treino e teste no formato HuggingFace
test_dataset = Dataset.from_dict({'content': test_texts, 'labels': test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True) # Aplica a tokenização aos datasets de treino e teste
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['content']) # Remover a coluna 'content' após a tokenização
test_dataset = test_dataset.remove_columns(['content'])

# AJUSTE FINO
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels']) # Definir os formatos de tensor (PyTorch)
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2) # Carregar o modelo DistilBERT para classificação

training_args = TrainingArguments( # Define os parâmetros de treinamento
    output_dir='./results',
    evaluation_strategy="epoch",
    logging_dir='./logs',
    learning_rate=3e-5, # Ajuste o learning rate
    num_train_epochs=20, # Quantidade de epocas. Experimentar entre 5 e 50 epocas
    per_device_train_batch_size=32, # Tamanho lote treino
    per_device_eval_batch_size=32, # Tamanho lote avaliação
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    eval_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=MetricasPersonalizadas  # Métricas personalizada
)

trainer.train()

# APRESENTAÇÃO E COMPARAÇÃO DOS RESULTADOS
results = trainer.evaluate()
print(f"\n\nResultados: {results}")
