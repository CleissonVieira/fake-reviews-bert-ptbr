from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import Dataset
import pandas as pd
from datetime import datetime
import numpy as np
import shutil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

url_dataset = 'https://raw.githubusercontent.com/lucaspercisi/yelp-fake-reviews-ptbr/main/Datasets/portuguese/yelp-fake-reviews-dataset-pt-pos-tagged.csv'
df = pd.read_csv(url_dataset)
df['cleaned_content'] = df['content']

df_falsos = df[df['fake_review'] == False].sample(3387, random_state=42)
df_verdadeiros = df[df['fake_review'] == True].sample(3387, random_state=42)
df_balanceado = pd.concat([df_falsos, df_verdadeiros]).sample(frac=1, random_state=42)

X = df_balanceado['cleaned_content']
y = df_balanceado['fake_review'].astype(int).values

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def tokenize_function(df):
    return tokenizer(df['text'], padding='max_length', truncation=True, max_length=256)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': round(acc, 5), 'precision': round(precision, 5), 'recall': round(recall, 5), 'f1': round(f1, 5) }

best_model_path = "../resultados/best_model_bert/"
_batch_size = 16

training_args = TrainingArguments(
    output_dir=best_model_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=_batch_size,
    per_device_eval_batch_size=_batch_size,
    num_train_epochs=15,
    weight_decay=0.03,
    learning_rate=5e-5,
    logging_dir=f'{best_model_path}logs',
    logging_steps=10,
    save_total_limit=1,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_eval_results = []

X = df_balanceado['cleaned_content']
y = df_balanceado['fake_review'].astype(int).values

#                                         best sample
split_train = 72 # 64    72,      81      72 
split_test = 0.2 # 0.2   0.1,     0.1     0.2
split_val = 8    # 16    18       9       8

X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=split_test, stratify=y, random_state=42)
data_hora = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
nome_arquivo_results = f"../resultados/resultadosTreinoTeste.csv"
nome_arquivo_log_validacao = f"../resultados/logTreino.csv"
observacao = 'df full 6774 balanceado, 15 epochs, 5 folds'

if True:
    for fold, (train_index, eval_index) in enumerate(skf.split(X_train_and_val, y_train_and_val)):
        print(f"\nTreinando o fold {fold + 1}/{k_folds}\n")
        
        # Dataset X_train_and_val dividido em: Treino 90%, Validação 10%
        half_eval_index = eval_index[:len(eval_index) // 2]
        train_index_with_half_eval = list(train_index) + list(half_eval_index)
        remaining_eval_index = eval_index[len(eval_index) // 2:]
        X_train, X_eval = X.iloc[train_index_with_half_eval], X.iloc[remaining_eval_index]
        y_train, y_eval = y[train_index_with_half_eval], y[remaining_eval_index]
        
        # Dataset X_train_and_val dividido em: Treino 80%, Validação 20%
        # X_train, X_eval = X.iloc[train_index], X.iloc[eval_index]
        # y_train, y_eval = y[train_index], y[eval_index]
        
        train_data = Dataset.from_dict({'text': X_train.tolist(), 'labels': y_train.tolist()})
        eval_data = Dataset.from_dict({'text': X_eval.tolist(), 'labels': y_eval.tolist()})
        
        # Tokenizando os dados
        train_data = train_data.map(tokenize_function, batched=True)
        eval_data = eval_data.map(tokenize_function, batched=True)
        
        # Inicializando o modelo para cada fold
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        eval_results = trainer.evaluate()
        fold_eval_results.append(eval_results)
        print(f"Resultados do fold {fold + 1}: {eval_results}")
        pd.DataFrame([{
            'data': data_hora,
            'log': f'{eval_results}',
        }]).round(5).to_csv(nome_arquivo_log_validacao, mode='a', header=False, index=False)

    model = BertForSequenceClassification.from_pretrained(trainer.state.best_model_checkpoint)
    model.save_pretrained(best_model_path)
    shutil.rmtree(trainer.state.best_model_checkpoint, ignore_errors=True)

    f1_scores = [result['eval_f1'] for result in fold_eval_results]
    print(f"\nF1 Score médio após cross-validation: {np.mean(f1_scores)}")

    pd.DataFrame([{
        'data': data_hora,
        'modo': 'train_VAL_test',
        'divisao': f'{split_train}_{split_val}_{split_test*100}',
        'classifier': 'BERT',
        'vectorizer': 'BERT',
        'features_used': '---',
        'accuracy_mean': np.mean([result['eval_accuracy'] for result in fold_eval_results]),
        'precision_mean': np.mean([result['eval_precision'] for result in fold_eval_results]),
        'recall_mean': np.mean([result['eval_recall'] for result in fold_eval_results]),
        'f1_score_mean': np.mean(f1_scores),
        'f1_score_variance': np.var(f1_scores),
        'f1_score_min': np.min(f1_scores),
        'f1_score_max': np.max(f1_scores),
        'observacao': observacao
    }]).round(5).to_csv(nome_arquivo_results, mode='a', header=False, index=False)


print("\nCarregando o melhor modelo para o teste final...")
model = BertForSequenceClassification.from_pretrained(best_model_path)
test_data = Dataset.from_dict({'text': X_test.tolist(), 'labels': y_test.tolist()})
test_data = test_data.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=best_model_path,
    per_device_eval_batch_size=_batch_size,
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
pd.DataFrame([{
    'data': data_hora,
    'modo': 'train_val_TEST',
    'divisao': f'{split_train}_{split_val}_{split_test*100}',
    'classifier': 'BERT',
    'vectorizer': 'BERT',
    'features_used': '---',
    'accuracy_mean': test_results['eval_accuracy'],
    'precision_mean': test_results['eval_precision'],
    'recall_mean': test_results['eval_recall'],
    'f1_score_mean': test_results['eval_f1'],
    'f1_score_variance': '---',
    'f1_score_min': '---',
    'f1_score_max': '---',
    'observacao': observacao
}]).round(5).to_csv(nome_arquivo_results, mode='a', header=False, index=False)

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