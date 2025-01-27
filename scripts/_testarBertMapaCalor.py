import os
import json
import shutil
import warnings
import itertools
import matplotlib # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns # type: ignore
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, f1_score
from transformers import     BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime
from datasets import Dataset

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

url_dataset = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/refs/heads/main/datasets/yelp-fake-reviews-dataset-pt.csv'
df = pd.read_csv(url_dataset)

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

vectorFeatures = [
    [f'content', df['content']],
    # [f'qtd_friends, qtd_reviews, qtd_photos. (sem legenda)', df.apply(lambda x: f"{x['qtd_friends']},{x['qtd_reviews']},{x['qtd_photos']}.", axis=1)],
    # [f'qtd_friends, qtd_reviews, qtd_photos. (com legenda)', df.apply(lambda x: f"Número de amigos: {x['qtd_friends']}, Número de avaliações: {x['qtd_reviews']}, Número de fotos: {x['qtd_photos']}.", axis=1)],
    # [f'content, qtd_friends, qtd_reviews, qtd_photos. (sem legenda)', df.apply(lambda x: f'"{x['content']}", {x['qtd_friends']}, {x['qtd_reviews']}, {x['qtd_photos']}.', axis=1)],
    # [f'content, qtd_friends, qtd_reviews, qtd_photos. (com legenda)', df.apply(lambda x: f'Review: "{x['content']}", Número de amigos: {x['qtd_friends']}, Número de avaliações: {x['qtd_reviews']}, Número de fotos: {x['qtd_photos']}.', axis=1)]
]

vectorLearningRate = [1e-5, 2.5e-5, 3e-5] # 1e-5, 2.5e-5, 2e-5, 3e-5
vectorWeightDecay = [0.001] # 0.05 0.03, 0.01, 0.001
combinations = itertools.product(vectorFeatures, vectorLearningRate, vectorWeightDecay)

_SAMPLE_DF = 3387
_MODELO = 'bert-base-multilingual-uncased'
_BATCH_SIZE = 16
_K_FOLDS = 3
_EPOCHS = 10
_EARLY_STOPPING = 4
_DROP_OUT=0.5
_MAX_LENGTH = 256
split_train = 64 
split_val = 16   
split_test = 0.2
patch = f'../resultados/'
patchModel = f'{patch}mapa_calor/'
nome_arquivo_results = f"resultadosMapaCalor.csv"
    
for feature, learning_rate, weight_decay in combinations:
    feature_name = feature[0]
    
    # if feature_name == 'content' and learning_rate in (1e-05, 2.5e-5, 3e-5) and weight_decay in (0.03, 0.01):
    #     continue
    
    print(f"Executando com: Learning Rate = {learning_rate}, Weight Decay = {weight_decay}, Feature = {feature_name}")
    
    _LEARNING_RATE = learning_rate
    _WEIGHT_DECAY=weight_decay
    _FEATURES = feature_name

    df['cleaned_content'] = feature[1]
    df_falsos = df[df['fake_review'] == False].sample(_SAMPLE_DF, random_state=42)
    df_verdadeiros = df[df['fake_review'] == True].sample(_SAMPLE_DF, random_state=42)
    df_balanceado = pd.concat([df_falsos, df_verdadeiros]).sample(frac=1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(_MODELO)
    skf = StratifiedKFold(n_splits=_K_FOLDS, shuffle=True, random_state=42)
    fold_eval_results = []

    X = df_balanceado['cleaned_content']
    y = df_balanceado['fake_review'].astype(int).values
    
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=split_test, stratify=y, random_state=42)

    data_hora = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    best_model_path = f"{patchModel}{data_hora}_{_MAX_LENGTH}_{_LEARNING_RATE}_{_WEIGHT_DECAY}/"
    observacao = f'df full {int(_SAMPLE_DF*2)} balanceado, {_EPOCHS} epochs, {_K_FOLDS} folds, early_stopping {_EARLY_STOPPING}, drop_out {_DROP_OUT}, batch_size {_BATCH_SIZE}'

    if best_model_path.__contains__("_256_1e-05_0.001"):
        best_model_path = f"{patchModel}20241111_220341_256_1e-05_0.001/"
    elif best_model_path.__contains__("_256_2.5e-05_0.001"):
        best_model_path = f"{patchModel}20241112_054859_256_2.5e-05_0.001/"
    elif best_model_path.__contains__("_256_3e-05_0.001"):
        best_model_path = f"{patchModel}20241112_125006_256_3e-05_0.001/"
    else:
        best_model_path = "deuruim/"
        
    print("\nCarregando o melhor modelo para o teste final...")
    model = BertForSequenceClassification.from_pretrained(best_model_path)
    
    f1_scores, accuracy_scores, precision_scores, recall_scores = [], [], [], []
    conf_matrix_total = np.zeros((2, 2), dtype=int)
    for seed in [1,9,29,42,60,100]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test, stratify=y, random_state=seed)
        
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
        f1_scores.append(test_results["eval_f1"])
        accuracy_scores.append(test_results["eval_accuracy"])
        precision_scores.append(test_results["eval_precision"])
        recall_scores.append(test_results["eval_recall"])
        
        test_predictions_logits = test_trainer.predict(test_data)
        test_predictions = np.argmax(test_predictions_logits.predictions, axis=1)
        
        conf_matrix = confusion_matrix(y_test, test_predictions)
        conf_matrix_total += conf_matrix

    SalvarCsv(patch, nome_arquivo_results, pd.DataFrame([{
        'date': data_hora,
        'mode': 'train_TEST',
        'division_dataset': f'{split_train}_{split_val}_{int(split_test*100)}',
        'model': 'BERT',
        'accuracy_mean': np.mean(accuracy_scores),
        'precision_mean': np.mean(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'f1_score_mean': np.mean(f1_scores),
        'f1_score_std': np.std(f1_scores),
        'f1_score_variance': np.var(f1_scores),
        'f1_score_min': np.min(f1_scores),
        'f1_score_max': np.max(f1_scores),
        'learning_rate': _LEARNING_RATE,
        'max_lenght': _MAX_LENGTH,
        'weight_decay': _WEIGHT_DECAY,
        'features_used': _FEATURES,
        'observation': observacao
    }]).round(5))

    log_history_path = os.path.join(best_model_path, f"log_history_test.json")
    with open(log_history_path, 'w') as f:
        json.dump(test_trainer.state.log_history, f, indent=4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Falso"], yticklabels=["Real", "Falso"])
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão - Conjunto de Teste")
    conf_matrix_path = f"{best_model_path}/confusion_matrix2.png"
    plt.savefig(conf_matrix_path)
    plt.show()
    plt.close()
