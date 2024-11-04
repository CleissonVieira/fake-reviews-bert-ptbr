import os
import json
import torch
import shutil
import warnings
import matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import tensorflow as tf
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from datasets import Dataset

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

url_dataset = 'https://raw.githubusercontent.com/lucaspercisi/yelp-fake-reviews-ptbr/main/Datasets/portuguese/yelp-fake-reviews-dataset-pt-pos-tagged.csv'
df = pd.read_csv(url_dataset)

_FEATURES = 'textual: content | numérica: qtd_friends, qtd_reviews, qtd_photos. (com legenda)'
df['cleaned_content'] = df['content']
df['features_numerics'] = df.apply(lambda x: [x['qtd_friends'], x['qtd_reviews'], x['qtd_photos']], axis=1)

_SAMPLE_DF = 3387

df_falsos = df[df['fake_review'] == False].sample(_SAMPLE_DF, random_state=42)
df_verdadeiros = df[df['fake_review'] == True].sample(_SAMPLE_DF, random_state=42)
df_balanceado = pd.concat([df_falsos, df_verdadeiros]).sample(frac=1, random_state=42)

_MODELO = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(_MODELO)

def SalvarCsv(pathCsv, nomeCsv, resultados):
    patch = f'{pathCsv}{nomeCsv}'
    if not os.path.exists(f'{pathCsv}'): os.makedirs(f'{pathCsv}')
    if os.path.isfile(f'{patch}'): resultados.to_csv(f'{patch}', mode='a', header=False, index=False)
    else: resultados.to_csv(f'{patch}', mode='w', index=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': round(acc, 5), 'precision': round(precision, 5), 'recall': round(recall, 5), 'f1': round(f1, 5) }

def tokenize_function(df):
    return tokenizer(df['text'], padding='max_length', truncation=True, max_length=_MAX_LENGTH)

_MAX_LENGTH = 256
_BATCH_SIZE = 16
_EPOCHS = 7
_LEARNING_RATE = 3e-5
_WEIGHT_DECAY=0.01
_EARLY_STOPPING = 2
_DROP_OUT=0.5
_K_FOLDS = 5
skf = StratifiedKFold(n_splits=_K_FOLDS, shuffle=True, random_state=42)
fold_eval_results = []

#                   best full                     best sample
split_train = 64 #  64           72,      81      72 
split_val = 16   #  16           18       9       8
split_test = 0.2 #  0.2          0.1,     0.1     0.2

X = df_balanceado['cleaned_content']
y = df_balanceado['fake_review'].astype(int).values

if 'features_numerics' in df.columns:
    _FEATURES_NUMERICS = True
    X_num = np.array(df_balanceado['features_numerics'].tolist())
    X_train_and_val, X_test, y_train_and_val, y_test, X_num_train_and_val, X_num_test = train_test_split(X, y, X_num, test_size=0.2, stratify=y, random_state=42)
else:
    _FEATURES_NUMERICS = False
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=split_test, stratify=y, random_state=42)

if _FEATURES_NUMERICS:
    # Normalizar os dados
    mean = X_num_train_and_val.mean(axis=0)
    std = X_num_train_and_val.std(axis=0)
    X_num_train_and_val = (X_num_train_and_val - mean) / std
    X_num_test = (X_num_test - mean) / std

    class BertWithNumericalFeatures(nn.Module):
        def __init__(self):
            super(BertWithNumericalFeatures, self).__init__()
            self.bert = BertModel.from_pretrained(_MODELO, ignore_mismatched_sizes=True)
            self.dropout = nn.Dropout(0.5)
            self.classifier = nn.Linear(768 + 3, 2)  # 768 hidden size do BERT + 3 features numéricas

        def save(self, path):
            self.bert.save_pretrained(path)
        
        def forward(self, input_ids, attention_mask, features_numerics=None, labels=None):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            if features_numerics is not None:
                combined_output = torch.cat((pooled_output, features_numerics), dim=1)
            else:
                combined_output = pooled_output

            combined_output = self.dropout(combined_output)
            logits = self.classifier(combined_output)

            # Calcule e retorne a perda, se os rótulos forem fornecidos
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
                return loss, logits  # Retorne a perda e os logits
            else:
                return logits  # Retorne apenas os logits se não houver rótulos

data_hora = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
patch = f'../resultados/'
patchModel = f'{patch}best_model_bert/'
best_model_path = f"{patchModel}{data_hora}/"
nome_arquivo_results = f"resultadosTreinoTeste.csv"
nome_arquivo_log_validacao = f"logTreino.csv"
observacao = f'df full {int(_SAMPLE_DF*2)} balanceado, {_EPOCHS} epochs, {_K_FOLDS} folds, lr {_LEARNING_RATE}, max_lenght {_MAX_LENGTH}, weight_decay {_WEIGHT_DECAY}, early_stopping {_EARLY_STOPPING}'

training_args = TrainingArguments(
    output_dir=best_model_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=_BATCH_SIZE,
    per_device_eval_batch_size=_BATCH_SIZE,
    num_train_epochs=_EPOCHS,
    weight_decay=_WEIGHT_DECAY,
    learning_rate=_LEARNING_RATE,
    logging_dir=f'{best_model_path}logs',
    logging_steps=10,
    save_total_limit=1,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)

if True:
    val_losses_per_fold = []
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=_EARLY_STOPPING)

    for fold, (train_index, eval_index) in enumerate(skf.split(X_train_and_val, y_train_and_val)):
        print(f"\nTreinando o fold {fold + 1}/{_K_FOLDS}\n")
        
        # Dataset X_train_and_val dividido em: Treino 80%, Validação 20%
        X_train, X_eval = X.iloc[train_index], X.iloc[eval_index]
        y_train, y_eval = y[train_index], y[eval_index]
        if _FEATURES_NUMERICS:
            X_num_train, X_num_eval = X_num_train_and_val[train_index], X_num_train_and_val[eval_index]
        
        train_data = Dataset.from_dict({'text': X_train.tolist(), 'labels': y_train.tolist()})
        eval_data = Dataset.from_dict({'text': X_eval.tolist(), 'labels': y_eval.tolist()})
        
        train_data = train_data.map(tokenize_function, batched=True)
        eval_data = eval_data.map(tokenize_function, batched=True)
        
        if _FEATURES_NUMERICS:
            train_data = train_data.add_column('features_numerics', X_num_train.tolist())
            eval_data = eval_data.add_column('features_numerics', X_num_eval.tolist())
            model = BertWithNumericalFeatures()
        else:
            model = BertForSequenceClassification.from_pretrained(_MODELO, num_labels=2)
            
        model.classifier.dropout = tf.keras.layers.Dropout(_DROP_OUT)
        
        trainer = Trainer(
            callbacks=[early_stopping_callback],
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics
        )
        trainer.train()
        
        log_history_path = os.path.join(best_model_path, f"log_history_fold_{fold + 1}.json")
        with open(log_history_path, 'w') as f:
            json.dump(trainer.state.log_history, f, indent=4)
        
        # Avaliar o *fold* e salvar o *loss* de validação de cada época
        for i, log in enumerate(trainer.state.log_history):
            if "eval_loss" in log and "epoch" in log:
                val_losses_per_fold.append(log["eval_loss"])
                print(f"Época {i+1}, *loss* de validação: {log['eval_loss']}")
        
        # Plotar o *loss* de validação para o *fold* atual
        plt.plot(val_losses_per_fold, label=f"Fold {fold + 1}")
        # Limpar os dados do *loss* para o próximo *fold*
        val_losses_per_fold.clear()
        
        eval_results = trainer.evaluate()
        fold_eval_results.append(eval_results)
        print(f"Resultados do fold {fold + 1}: {eval_results}")
        
        SalvarCsv(best_model_path, nome_arquivo_log_validacao, pd.DataFrame([{
            'tipo': 'treino',
            'data': data_hora,
            'log': f'{eval_results}',
        }]).round(5))
        
    _MODELO = trainer.state.best_model_checkpoint
    
    if _FEATURES_NUMERICS:
        model = BertWithNumericalFeatures().save(best_model_path)
    else:
        model = BertForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint)
        model.save_pretrained(best_model_path)
    
    shutil.rmtree(trainer.state.best_model_checkpoint, ignore_errors=True)

    plt.xlabel("Época")
    plt.ylabel("Loss de Validação")
    plt.title("Loss de Validação por Época em Cada Fold")
    plt.legend()
    plt.savefig(f"{best_model_path}loss_validacao_folds.png", format="png")

    f1_scores = [result['eval_f1'] for result in fold_eval_results]
    print(f"\nF1 Score médio após cross-validation: {np.mean(f1_scores)}")

    SalvarCsv(patch, nome_arquivo_results, pd.DataFrame([{
        'data': data_hora,
        'modo': 'train_VAL_test',
        'divisao': f'{split_train}_{split_val}_{int(split_test*100)}',
        'classifier': 'BERT',
        'vectorizer': 'BERT',
        'features_used': _FEATURES,
        'accuracy_mean': np.mean([result['eval_accuracy'] for result in fold_eval_results]),
        'precision_mean': np.mean([result['eval_precision'] for result in fold_eval_results]),
        'recall_mean': np.mean([result['eval_recall'] for result in fold_eval_results]),
        'f1_score_mean': np.mean(f1_scores),
        'f1_score_variance': np.var(f1_scores),
        'f1_score_min': np.min(f1_scores),
        'f1_score_max': np.max(f1_scores),
        'observacao': observacao
    }]).round(5))
    

print("\nCarregando o melhor modelo para o teste final...")

_MODELO = best_model_path

if _FEATURES_NUMERICS:
    model = BertWithNumericalFeatures()
else:
    model = BertForSequenceClassification.from_pretrained(best_model_path)

test_data = Dataset.from_dict({'text': X_test.tolist(), 'labels': y_test.tolist()})
test_data = test_data.map(tokenize_function, batched=True)

if _FEATURES_NUMERICS:
    test_data = test_data.add_column('features_numerics', X_num_test.tolist())

training_args = TrainingArguments(
    output_dir=best_model_path,
    per_device_eval_batch_size=_BATCH_SIZE,
    metric_for_best_model="f1",
)

test_trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

test_results = test_trainer.evaluate(eval_dataset=test_data)
print(f"Resultados teste: {test_results}")
print(f"\nF1 Score médio após teste: {test_results['eval_f1']}")

SalvarCsv(patch, nome_arquivo_results, pd.DataFrame([{
        'data': data_hora,
        'modo': 'train_val_TEST',
        'divisao': f'{split_train}_{split_val}_{int(split_test*100)}',
        'classifier': 'BERT',
        'vectorizer': 'BERT',
        'features_used': _FEATURES,
        'accuracy_mean': test_results['eval_accuracy'],
        'precision_mean': test_results['eval_precision'],
        'recall_mean': test_results['eval_recall'],
        'f1_score_mean': test_results['eval_f1'],
        'f1_score_variance': '---',
        'f1_score_min': '---',
        'f1_score_max': '---',
        'observacao': observacao
    }]).round(5))

# Previsões no conjunto de teste
test_predictions_logits = test_trainer.predict(test_data)
test_predictions = np.argmax(test_predictions_logits.predictions, axis=1)

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test, test_predictions)

# Plotar e salvar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Falso"], yticklabels=["Real", "Falso"])
plt.xlabel("Predição")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão - Conjunto de Teste")
conf_matrix_path = f"{best_model_path}/confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.show()