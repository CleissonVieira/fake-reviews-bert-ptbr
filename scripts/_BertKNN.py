from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import re
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate

nltk.download('stopwords')
stop_words_pt = set(stopwords.words('portuguese'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words_pt])
    return text

url_dataset = 'https://raw.githubusercontent.com/lucaspercisi/yelp-fake-reviews-ptbr/main/Datasets/portuguese/yelp-fake-reviews-dataset-pt-pos-tagged.csv'
yelp_df = pd.read_csv(url_dataset)

# Limpando conteúdo textual
yelp_df['cleaned_content'] = yelp_df['content'] #.apply(clean_text)

# Balanceando o dataset
df_falsos = yelp_df[yelp_df['fake_review'] == False]
df_verdadeiros = yelp_df[yelp_df['fake_review'] == True]
df_falsos = df_falsos.sample(3387, random_state=42)
df_verdadeiros = df_verdadeiros.sample(3387, random_state=42)
yelp_df_balanceado = pd.concat([df_falsos, df_verdadeiros])
yelp_df_sample = yelp_df_balanceado.copy()
X = yelp_df_sample['cleaned_content']
y = yelp_df_sample['fake_review'].values

# Definindo a função para rodar o classificador e salvar os resultados
def run_and_save_results(clf, X, y, classifier_name, vectorizer, results_df, features_useds):
    scorers = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1_score': 'f1'
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scorers, return_train_score=False)

    features_used = ', '.join(features_useds.columns) if not features_useds.empty else '---'
    results = {
        'data': f'{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'modo': 'train_test',
        'divisao': '80_20',
        'classifier': classifier_name,
        'vectorizer': vectorizer,
        'features_used': features_used,
        'accuracy_mean': np.mean(cv_results['test_accuracy']),
        'precision_mean': np.mean(cv_results['test_precision']),
        'recall_mean': np.mean(cv_results['test_recall']),
        'f1_score_mean': np.mean(cv_results['test_f1_score']),
        'f1_score_variance': np.var(cv_results['test_f1_score'], ddof=1),
        'f1_score_min': np.min(cv_results['test_f1_score']),
        'f1_score_max': np.max(cv_results['test_f1_score']),
        'observacao': ''
    }

    updated_results_df = pd.concat([results_df, pd.DataFrame([results]).round(5)], axis=0, ignore_index=True)
    nome_arquivo = f"../resultados/resultadosTreinoTeste.csv"
    updated_results_df.to_csv(nome_arquivo, mode='a', header=False, index=False)
    return updated_results_df

# Classe para usar BERT como vetorizador
class BERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-multilingual-uncased', max_length=256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.model.half()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = []
        for text in X:
            inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embedding
            embeddings.append(cls_embedding)
        return np.vstack(embeddings)

results_df_global = pd.DataFrame(columns=[
    'data', 'modo', 'classifier', 'vectorizer', 'features_used', 'accuracy_mean',
    'precision_mean','recall_mean', 'f1_score_mean', 'f1_score_variance', 'f1_score_min', 'f1_score_max'
])

# Rodando com BERT
bert_vect = BERTVectorizer()
X_bert_transformed = bert_vect.transform(X)

clf_name = 'KNN'
classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=17, metric='euclidean', weights='distance')
print(f"Iniciando BERT, {clf_name}")

# Adicionando as features numéricas se necessário
colunas_a_incluir = None # ['qtd_friends', 'qtd_reviews', 'qtd_photos']
X_numeric = yelp_df_sample[colunas_a_incluir].values if colunas_a_incluir else None
X_combined = np.hstack((X_bert_transformed, X_numeric)) if colunas_a_incluir is not None else X_bert_transformed

# Executando o classificador e salvando os resultados
features_used = pd.DataFrame(X_numeric, columns=colunas_a_incluir) if X_numeric is not None else pd.DataFrame()
results_df_global = run_and_save_results(classifier, X_combined, y, clf_name, 'BERT', results_df_global, features_used)

f1_score_atual = results_df_global[(results_df_global['classifier'] == clf_name) & (results_df_global['vectorizer'] == 'BERT')]['f1_score_mean'].iloc[-1]
print(f"F1 Score para BERT e {clf_name}: {f1_score_atual}")
