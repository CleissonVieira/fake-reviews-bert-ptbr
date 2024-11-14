import os
import json
import shutil
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from datasets import Dataset

import subprocess

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

url_dataset = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/refs/heads/main/datasets/yelp-fake-reviews-dataset-pt.csv'
df = pd.read_csv(url_dataset)

_MODELO = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(_MODELO)

def SalvarCsv(pathCsv, nomeCsv, resultados):
    patch = f'{pathCsv}{nomeCsv}'
    if not os.path.exists(f'{pathCsv}'): os.makedirs(f'{pathCsv}')
    if os.path.isfile(f'{patch}'): resultados.to_csv(f'{patch}', mode='a', header=False, index=False)
    else: resultados.to_csv(f'{patch}', mode='w', index=False)

NUM_BOOTSTRAP_SAMPLES = 50
def calculate_bootstrap_metrics(trainer, test_data, n_bootstraps=NUM_BOOTSTRAP_SAMPLES):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(n_bootstraps):
        # Cria uma amostra de bootstrap com reposição
        bootstrap_indices = np.random.choice(len(test_data), len(test_data), replace=True)
        bootstrap_sample = test_data.select(bootstrap_indices)

        # Avalia a amostra com o modelo
        predictions = trainer.predict(bootstrap_sample)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    metrics_results = {
        'accuracy_mean': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'f1_score_mean': np.mean(f1_scores),
        'f1_variance': np.var(f1_scores),
        'f1_score_std': np.std(f1_scores),
        'f1_score_min': np.min(f1_scores),
        'f1_score_max': np.max(f1_scores),
    }
    
    SalvarCsv(patch, nome_arquivo_results, pd.DataFrame([{
        'date': data_hora,
        'mode': 'train_val_TEST',
        'division_dataset': f'{split_train}_{split_val}_{int(split_test*100)}',
        'classifier': 'BERT',
        'vectorizer': 'BERT',
        'features_used': _FEATURES,
        'accuracy_mean': np.mean(accuracy_scores),
        'precision_mean': np.mean(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'f1_score_mean': np.mean(f1_scores),
        'f1_score_std': np.std(f1_scores),
        'f1_score_variance': np.var(f1_scores),
        'f1_score_min': np.min(f1_scores),
        'f1_score_max': np.max(f1_scores),
        'observation': observacao
    }]).round(5))
    
    return metrics_results

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
# _EPOCHS = 7
# _LEARNING_RATE = 2.5e-5
# _WEIGHT_DECAY=0.01
# _EARLY_STOPPING = 2
# _DROP_OUT=0.5
_K_FOLDS = 5
skf = StratifiedKFold(n_splits=_K_FOLDS, shuffle=True, random_state=42)
fold_eval_results = []

#                   best full                           best sample 1000
split_train = 64 #  64           72,        81          72 
split_val = 16   #  16           18         9           8
split_test = 0.2 #  0.2*100      0.1*100,   0.1*100     0.2*100


for pasta in ["numerics_20241103_142000"]: # "content_20241030_165616", "numerics_20241103_142000", "contentNumerics_20241103_184343"
    if pasta == "content_20241030_165616":
        _FEATURES = f'textual: content'
        df['cleaned_content'] = df['content']
    
    elif pasta == "numerics_20241103_142000":
        _FEATURES = f'numTextual: qtd_friends, qtd_reviews, qtd_photos. (sem legenda)'
        df['cleaned_content'] = df.apply(lambda x: f"Número de amigos: {x['qtd_friends']}. Número de avaliações: {x['qtd_reviews']}. Número de fotos: {x['qtd_photos']}.", axis=1)


    elif pasta == "contentNumerics_20241103_184343":
        _FEATURES = f'textual+numTextual: content, qtd_friends, qtd_reviews, qtd_photos. (sem legenda, com aspas)'
        df['cleaned_content'] = df.apply(lambda x: f'"{x['content']}", {x['qtd_friends']}, {x['qtd_reviews']}, {x['qtd_photos']}.', axis=1)
    

    _SAMPLE_DF = 3387
    df_falsos = df[df['fake_review'] == False].sample(_SAMPLE_DF, random_state=42)
    df_verdadeiros = df[df['fake_review'] == True].sample(_SAMPLE_DF, random_state=42)
    df_balanceado = pd.concat([df_falsos, df_verdadeiros]).sample(frac=1, random_state=42)

    X = df_balanceado['cleaned_content']
    y = df_balanceado['fake_review'].astype(int).values

    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=split_test, stratify=y, random_state=42)

    data_hora = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    patch = f'../resultados/'
    patchModel = f'{patch}best_model_bert/'
    best_model_path = f"{patchModel}{pasta}/"
    nome_arquivo_results = f"resultadosTreinoTeste.csv"
    nome_arquivo_log_validacao = f"logTreino.csv"
    observacao =  "" #f'df full {int(_SAMPLE_DF*2)} balanceado, {_EPOCHS} epochs, {_K_FOLDS} folds, lr {_LEARNING_RATE}, max_lenght {_MAX_LENGTH}, weight_decay {_WEIGHT_DECAY}, early_stopping {_EARLY_STOPPING}, drop_out {_DROP_OUT}'

    print("\nCarregando o melhor modelo para o teste final...")
    model = BertForSequenceClassification.from_pretrained(best_model_path)
    test_data = Dataset.from_dict({'text': X_test.tolist(), 'labels': y_test.tolist()})
    test_data = test_data.map(tokenize_function, batched=True)

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

    log_history_path = os.path.join(best_model_path, f"log_history_test2.json")
    with open(log_history_path, 'w') as f:
        json.dump(test_trainer.state.log_history, f, indent=4)

    print(f"Resultados teste: {test_results}")
    print(f"\nF1 Score médio após teste: {test_results['eval_f1']}")

    SalvarCsv(patch, nome_arquivo_results, pd.DataFrame([{
            'date': data_hora,
            'mode': 'train_val_TEST',
            'division_dataset': f'{split_train}_{split_val}_{int(split_test*100)}',
            'classifier': 'BERT',
            'vectorizer': 'BERT',
            'features_used': _FEATURES,
            'accuracy_mean': test_results['eval_accuracy'],
            'precision_mean': test_results['eval_precision'],
            'recall_mean': test_results['eval_recall'],
            'f1_score_mean': test_results['eval_f1'],
            'f1_score_std': '---',
            'f1_score_variance': '---',
            'f1_score_min': '---',
            'f1_score_max': '---',
            'observation': observacao
        }]).round(5))
    
    calculate_bootstrap_metrics(test_trainer, test_data)

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
    conf_matrix_path = f"{best_model_path}/confusion_matrix2.png"
    plt.savefig(conf_matrix_path)
    plt.show()
    
    
subprocess.run(["python", "_Bert.py"])