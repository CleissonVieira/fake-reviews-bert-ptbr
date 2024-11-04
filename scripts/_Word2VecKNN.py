
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

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
yelp_df['cleaned_content'] = yelp_df['content'].apply(clean_text)

df_falsos = yelp_df[yelp_df['fake_review'] == False]
df_verdadeiros = yelp_df[yelp_df['fake_review'] == True]
df_falsos = df_falsos.sample(3387, random_state=42)
df_verdadeiros = df_verdadeiros.sample(3387, random_state=42)
yelp_df_balanceado = pd.concat([df_falsos, df_verdadeiros])
yelp_df_sample = yelp_df_balanceado.copy()
X = yelp_df_sample['cleaned_content']
y = yelp_df_sample['fake_review'].values

# X_train_and_val = X
# y_train_and_val = y

split_train = 64 #  64           72,      81      72 
split_val = 16   #  16           18       9       8
split_test = 0.2 #  0.2          0.1,     0.1     0.2
X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=split_test, stratify=y, random_state=42)

# Função para executar o classificador com um número reduzido de features e salvar os resultados
def eval_run_and_save_results(clf, Xs, ys, classifier_name, vectorizer, results_df, features_useds):
    scorers = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1_score': 'f1'
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, Xs, ys, cv=cv, scoring=scorers, return_train_score=False, verbose=3)

    features_used = ', '.join(features_useds.columns) if not features_useds.empty else '---'
    results = {
        'data': f'{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'modo': 'train_VAL_test',
        'divisao': f'{split_train}_{split_val}_{int(split_test*100)}',
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
        'observacao': 'df full 6774 balanceado, acrescentado teste fora dos folds'
    }

    updated_results_df = pd.concat([results_df, pd.DataFrame([results]).round(5)], axis=0, ignore_index=True)
    nome_arquivo = f"../resultados/resultadosTreinoTeste.csv"
    updated_results_df.to_csv(nome_arquivo, mode='a', header=False, index=False)
    
    return updated_results_df, clf

def test_run_and_save_results(clf, Xs, ys, classifier_name, vectorizer, results_df, features_useds):
    # clf = joblib.load(f'../resultados/best_model_w2vecKnn/w2vecKnn.joblib')
    
    y_pred = clf.predict(Xs)

    features_used = ', '.join(features_useds.columns) if not features_useds.empty else '---'
    results = {
        'data': f'{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'modo': 'train_val_TEST',
        'divisao': f'{split_train}_{split_val}_{int(split_test*100)}',
        'classifier': classifier_name,
        'vectorizer': vectorizer,
        'features_used': features_used,
        'accuracy_mean': accuracy_score(y_test, y_pred),
        'precision_mean': precision_score(y_test, y_pred),
        'recall_mean': recall_score(y_test, y_pred),
        'f1_score_mean': f1_score(y_test, y_pred),
        'f1_score_variance': '---',
        'f1_score_min': '---',
        'f1_score_max': '---',
        'observacao': 'df full 6774 balanceado, acrescentado teste fora dos folds'
    }

    updated_results_df = pd.concat([results_df, pd.DataFrame([results]).round(5)], axis=0, ignore_index=True)
    nome_arquivo = f"../resultados/resultadosTreinoTeste.csv"
    updated_results_df.to_csv(nome_arquivo, mode='a', header=False, index=False)
    
    return updated_results_df

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.word2vec = None
        self.dim = None

    def fit(self, Xs, y=None):
        sentences = [sentence.split() for sentence in Xs]
        self.word2vec = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count)
        self.dim = self.word2vec.vector_size
        return self

    def transform(self, Xs):
        return np.array([
            np.mean([self.word2vec.wv[word] for word in words if word in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in [sentence.split() for sentence in Xs]
        ])
   
val_results_df_global = pd.DataFrame(columns=[
    'data', 'modo', 'divisao', 'classifier', 'vectorizer', 'features_used', 'accuracy_mean',
    'precision_mean','recall_mean', 'f1_score_mean', 'f1_score_variance', 'f1_score_min', 'f1_score_max'
])

test_results_df_global = pd.DataFrame(columns=[
    'data', 'modo', 'divisao', 'classifier', 'vectorizer', 'features_used', 'accuracy_mean',
    'precision_mean','recall_mean', 'f1_score_mean', 'f1_score_variance', 'f1_score_min', 'f1_score_max'
])
    
clf_name = 'KNN' 
classifier =  KNeighborsClassifier(n_jobs=-1, n_neighbors=17, metric='euclidean', weights='distance')
print(f"Iniciando Word2Vec, {clf_name}")
    
# Rodando Word2Vec (se quiser utilizar feature textual)
w2v_vect = Word2VecVectorizer(vector_size=100, window=5, min_count=1)
w2v_vect.fit(X_train_and_val)
X_eval_transformed = w2v_vect.transform(X_train_and_val)
X_test_transformed = w2v_vect.transform(X_test)

# Escolhendo as features numéricas apropriadas
colunas_a_incluir = ['qtd_friends', 'qtd_reviews', 'qtd_photos'] # None 
X_val_numeric, X_test_numeric = None, None
if colunas_a_incluir is not None:
    X_val_numeric, X_test_numeric = train_test_split(yelp_df_sample[colunas_a_incluir].values, test_size=split_test, stratify=y, random_state=42)

# 100% dataset treino e teste
# X_val_numeric = yelp_df_sample[colunas_a_incluir].values if colunas_a_incluir else None

# content
# X_eval_combined = X_eval_transformed
# X_test_combined = X_test_transformed
# numéricas
# X_eval_combined = X_val_numeric
# X_test_combined = X_test_numeric
# content e numéricas
X_eval_combined = np.hstack((X_eval_transformed, X_val_numeric)) if colunas_a_incluir is not None else X_eval_transformed
X_test_combined = np.hstack((X_test_transformed, X_test_numeric)) if colunas_a_incluir is not None else X_test_transformed

# Executando o classificador e salvando os resultados
val_features_used = pd.DataFrame(X_val_numeric, columns=colunas_a_incluir)
val_results_df_global, classifier = eval_run_and_save_results(classifier, X_eval_combined, y_train_and_val, clf_name, 'Word2Vec', val_results_df_global, val_features_used)
f1_score_atual = val_results_df_global[(val_results_df_global['classifier'] == clf_name) & (val_results_df_global['vectorizer'] == 'Word2Vec')]['f1_score_mean'].iloc[-1]
print(f"Validação: F1 Score para Word2Vec e {clf_name}: {f1_score_atual}")

#Ajustando o classificador
classifier.fit(X_eval_combined, y_train_and_val)
# joblib.dump(classifier, f'../resultados/best_model_w2vecKnn/w2vecKnn.joblib')

test_features_used = pd.DataFrame(X_test_numeric, columns=colunas_a_incluir)
test_results_df_global = test_run_and_save_results(classifier, X_test_combined, y_test, clf_name, 'Word2Vec', test_results_df_global, test_features_used)
f1_score_atual = test_results_df_global[(test_results_df_global['classifier'] == clf_name) & (test_results_df_global['vectorizer'] == 'Word2Vec')]['f1_score_mean'].iloc[-1]
print(f"Teste: F1 Score para Word2Vec e {clf_name}: {f1_score_atual}")
