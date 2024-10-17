import numpy as np
import os
import pandas as pd
import time
import torch
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from _Utils import Utils

warnings.filterwarnings('ignore')
Utils.set_seed(42)

urlPtBr = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
dfPtBr = pd.read_csv(urlPtBr)
dfPtBr = dfPtBr[['content', 'fake_review']]

RELATORIO_GERAL = 'RelatorioTreinoValidacao.csv'
TOKENIZADOR = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 50
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 5e-5

class DatasetBert(Dataset):
    def __init__(self, encodings): self.encodings = encodings
    def __getitem__(self, idx): return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self): return len(self.encodings['input_ids'])

def TreinarModelo(modelo, train_loader, otimizador, agendador):
    modelo.train()
    perdaTotal = 0

    for batch in train_loader:
        otimizador.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        saidas = modelo(**batch)
        perda = saidas.loss
        perdaTotal += perda.item()
        perda.backward()
        otimizador.step()
        agendador.step()

    torch.cuda.empty_cache()
    return perdaTotal / len(train_loader)

def AvaliarModelo(modelo, valLoader, nomeDf, nomePng):
    modelo.eval()
    preds, trueLabels = [], []

    with torch.no_grad():
        for batch in valLoader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            saidas = modelo(**batch)
            logits = saidas.logits
            preds.extend(torch.argmax(logits, axis=-1).cpu().numpy())
            trueLabels.extend(batch['labels'].cpu().numpy())

    report = classification_report(trueLabels, preds, target_names=['Real', 'Fake'], output_dict=True)
    dfClassificationReport = pd.DataFrame(report).transpose()
    dfTitulo = pd.DataFrame([[f'Classification Report ({nomeDf} - {datetime.now().strftime("%Y-%m-%d_%H-%M")}):']])
    file_exists = os.path.isfile(f'{PATH_DATA_EXECUCAO}relatorioMatrizConfusao.csv')
    
    with open(f'{PATH_DATA_EXECUCAO}relatorioMatrizConfusao.csv', 'a') as f:
        dfTitulo.to_csv(f, index=False, header=not file_exists)
        dfClassificationReport.to_csv(f, index=True, header=True) 
        f.write('\n')
    
    Utils.GerarMatrizConfusao(trueLabels, preds, ['Real', 'Fake'], PATH_DATA_EXECUCAO, nomeDf, nomePng)

    precision = precision_score(trueLabels, preds)
    recall, f1 = recall_score(trueLabels, preds), f1_score(trueLabels, preds)

    precision_real, precision_fake = precision_score(trueLabels, preds, pos_label=0), precision_score(trueLabels, preds, pos_label=1)
    recall_real, recall_fake = recall_score(trueLabels, preds, pos_label=0), recall_score(trueLabels, preds, pos_label=1)
    f1_real, f1_fake = f1_score(trueLabels, preds, pos_label=0), f1_score(trueLabels, preds, pos_label=1)

    return precision, recall, f1, precision_real, precision_fake, recall_real, recall_fake, f1_real, f1_fake



# MÉTODO PRINCIPAL

def IniciarProcesso(dfUtilizado, conjuntoDados, nomeDf):
    print(f'\nLR {LEARNING_RATE} | EP {EPOCHS} | Dataset {conjuntoDados} | Verdadeiros: {len(dfUtilizado[dfUtilizado.fake_review == False])} | Falsos: {len(dfUtilizado[dfUtilizado.fake_review == True])}')
    if not os.path.exists(f'{PATH_DATA_EXECUCAO}{nomeDf}/'): os.makedirs(f'{PATH_DATA_EXECUCAO}{nomeDf}/')

    temporizador, dataStart = time.time(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    
    metricasValidacao = {
        'Precision_Real': [], 'Precision_Fake': [], 'Precision': [],
        'F1_Score_Real': [], 'F1_Score_Fake': [], 'F1_Score': [],
        'Recall_Real': [], 'Recall_Fake': [], 'Recall': [],
        'tempoTreino': 0, 'tempoValidacao': 0
    }
    
    metricasTeste = {
        'Precision_Real': [], 'Precision_Fake': [], 'Precision': [],
        'F1_Score_Real': [], 'F1_Score_Fake': [], 'F1_Score': [],
        'Recall_Real': [], 'Recall_Fake': [], 'Recall': []
    }
    
    encodingsCompleto = Utils.TokenizarTextos(dfUtilizado['content'].values, dfUtilizado['fake_review'].values, TOKENIZADOR, MAX_LENGTH)
    
    conteudoAvaliacao, booleanoRealFake = dfUtilizado['content'].values, dfUtilizado['fake_review'].values
    
    for kfolds, (treino80, teste20) in enumerate(skf.split(conteudoAvaliacao, booleanoRealFake)):
        print(f"\nDobras {kfolds + 1}/5")
    
        conteudoTreino60, conteudoTeste = conteudoAvaliacao[treino80], conteudoAvaliacao[teste20]
        realFakeTreino60, realFakeTeste = booleanoRealFake[treino80], booleanoRealFake[teste20]
    
        conteudoTreino60, conteudoValidacao20, realFakeTreino60, realFakeValidacao20 = train_test_split(
            conteudoTreino60, realFakeTreino60, test_size=0.25, stratify=realFakeTreino60, random_state=42
        )
    
        encodingsTreino = Utils.TokenizarTextos(conteudoTreino60, realFakeTreino60, TOKENIZADOR, MAX_LENGTH)
        encodingsValidacao = Utils.TokenizarTextos(conteudoValidacao20, realFakeValidacao20, TOKENIZADOR, MAX_LENGTH)
        encodingsTeste = Utils.TokenizarTextos(conteudoTeste, realFakeTeste, TOKENIZADOR, MAX_LENGTH)
        
        datasetTreino = DatasetBert(encodingsTreino) 
        datasetValidacao = DatasetBert(encodingsValidacao)
        datasetTeste = DatasetBert(encodingsTeste)
        
        dataLoaderTreino = DataLoader(datasetTreino, batch_size=BATCH_SIZE, shuffle=True)
        dataLoaderValidacao = DataLoader(datasetValidacao, batch_size=BATCH_SIZE, shuffle=False)
        dataLoaderTeste = DataLoader(datasetTeste, batch_size=BATCH_SIZE, shuffle=False)
    
        modeloUtilizado = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
        modeloUtilizado.to(DEVICE)
    
        otimizador, agendador = Utils.ConfigurarOtimizadorPerdaAgendador(modeloUtilizado, dataLoaderTreino, LEARNING_RATE, EPOCHS)
    
        for ep in range(0, EPOCHS, 5):
            print(f"Treinando Epoch de {ep + 1} até {ep + 5} | Memória GPU alocada: {torch.cuda.memory_allocated()/1024/1024} MB")
            temporizador = time.time()

            for epoch in range(5):
                perdaTreino = TreinarModelo(modeloUtilizado, dataLoaderTreino, otimizador, agendador)
                print(f"Epoch {epoch + 1}, Loss: {perdaTreino:.4f}")
            temporizadorTreino = time.time() - temporizador

            print(f"Avaliação intervalo {ep+1} até {ep+5} épocas")
            nomePng = f'Val_Epoch-{ep+1}a{ep+5}_Dobra-{kfolds+1}'
            temporizador = time.time()
            precision, recall, f1, precisionReal, precisionFake, recallReal, recallFake, f1Real, f1Fake = AvaliarModelo(modeloUtilizado, dataLoaderValidacao, nomeDf, nomePng)
            temporizadorValidacao = time.time() - temporizador
        
            metricasValidacao['tempoTreino'] += temporizadorTreino
            metricasValidacao['tempoValidacao'] += temporizadorValidacao
            metricasValidacao['Precision_Real'].append(precisionReal)
            metricasValidacao['Precision_Fake'].append(precisionFake)
            metricasValidacao['Recall_Real'].append(recallReal)
            metricasValidacao['Recall_Fake'].append(recallFake)
            metricasValidacao['F1_Score_Real'].append(f1Real)
            metricasValidacao['F1_Score_Fake'].append(f1Fake)
            metricasValidacao['Precision'].append(precision)
            metricasValidacao['Recall'].append(recall)
            metricasValidacao['F1_Score'].append(f1)
        
        # entender se é para substituir as métricas adicionadas da validação pelas de teste
        # se eu retorno e salvo o modelo treinado, salvo do último kfold ou misturo os resultados para formar o modelo
        
        print(f"Avaliação no conjunto de teste")
        nomePng = f'Test_Epoch-{EPOCHS}_Dobra-{kfolds+1}'
        precision_teste, recall_teste, f1_teste, precisionReal_teste, precisionFake_teste, recallReal_teste, recallFake_teste, f1Real_teste, f1Fake_teste = AvaliarModelo(modeloUtilizado, dataLoaderTeste, nomeDf, nomePng)
        
        metricasTeste['Precision_Real'].append(precisionReal_teste)
        metricasTeste['Precision_Fake'].append(precisionFake_teste)
        metricasTeste['Recall_Real'].append(recallReal_teste)
        metricasTeste['Recall_Fake'].append(recallFake_teste)
        metricasTeste['F1_Score_Real'].append(f1Real_teste)
        metricasTeste['F1_Score_Fake'].append(f1Fake_teste)
        metricasTeste['Precision'].append(precision_teste)
        metricasTeste['Recall'].append(recall_teste)
        metricasTeste['F1_Score'].append(f1_teste)
        
        resultadosDf = pd.DataFrame([{
            'scenario': 'Review',
            'classifier': 'bert-base-multilingual-uncased',
            'features_used': 'content',
            'data_execucao': dataStart,
            'tempo_treino_segundos': metricasValidacao['tempoTreino'],
            'tempo_validacao_segundos': metricasValidacao['tempoValidacao'],
            'max_length_tokenizer': MAX_LENGTH,
            'learning_rate': LEARNING_RATE,
            'total_avaliacoes_dataLoader': len(dfUtilizado),
            'total_avaliacoes_verdadeiras': len(dfUtilizado[dfUtilizado.fake_review == False]),
            'total_avaliacoes_falsas': len(dfUtilizado[dfUtilizado.fake_review == True]),
            'dataset': conjuntoDados,
            'dobra': kfolds + 1,
            'epocas': EPOCHS,
            'val_f1_score_real': np.mean(metricasValidacao['F1_Score_Real']),
            'test_f1_score_real': np.mean(metricasTeste['F1_Score_Real']),
            'val_f1_score_fake': np.mean(metricasValidacao['F1_Score_Fake']),
            'test_f1_score_fake': np.mean(metricasTeste['F1_Score_Fake']),
            'val_f1_score': np.mean(metricasValidacao['F1_Score']),
            'test_f1_score': np.mean(metricasTeste['F1_Score']),
            'val_f1_score_variance': np.var(metricasValidacao['F1_Score'], ddof=1),
            'test_f1_score_variance': np.var(metricasTeste['F1_Score'], ddof=1),
            'val_f1_score_min': np.min(metricasValidacao['F1_Score']),
            'test_f1_score_min': np.min(metricasTeste['F1_Score']),
            'val_f1_score_max': np.max(metricasValidacao['F1_Score']),
            'test_f1_score_max': np.max(metricasTeste['F1_Score']),
            'val_precision_real': np.mean(metricasValidacao['Precision_Real']),
            'test_precision_real': np.mean(metricasTeste['Precision_Real']),
            'val_precision_fake': np.mean(metricasValidacao['Precision_Fake']),
            'test_precision_fake': np.mean(metricasTeste['Precision_Fake']),
            'val_precision': np.mean(metricasValidacao['Precision']),
            'test_precision': np.mean(metricasTeste['Precision']),
            'val_precision_variance': np.var(metricasValidacao['Precision'], ddof=1),
            'test_precision_variance': np.var(metricasTeste['Precision'], ddof=1),
            'val_precision_min': np.min(metricasValidacao['Precision']),
            'test_precision_min': np.min(metricasTeste['Precision']),
            'val_precision_max': np.max(metricasValidacao['Precision']),
            'test_precision_max': np.max(metricasTeste['Precision']),
            'val_recall_real': np.mean(metricasValidacao['Recall_Real']),
            'test_recall_real': np.mean(metricasTeste['Recall_Real']),
            'val_recall_fake': np.mean(metricasValidacao['Recall_Fake']),
            'test_recall_fake': np.mean(metricasTeste['Recall_Fake']),
            'val_recall': np.mean(metricasValidacao['Recall']),
            'test_recall': np.mean(metricasTeste['Recall']),
            'val_recall_variance': np.var(metricasValidacao['Recall'], ddof=1),
            'test_recall_variance': np.var(metricasTeste['Recall'], ddof=1),
            'val_recall_min': np.min(metricasValidacao['Recall']),
            'test_recall_min': np.min(metricasTeste['Recall']),
            'val_recall_max': np.max(metricasValidacao['Recall']),
            'test_recall_max': np.max(metricasTeste['Recall'])
        }]).round(5)

        if os.path.isfile(RESULTADOS_CSV): resultadosDf.to_csv(RESULTADOS_CSV, mode='a', header=False, index=False)
        else: resultadosDf.to_csv(RESULTADOS_CSV, mode='w', index=False)
        
        torch.save(modeloUtilizado.state_dict(), f'{PATH_DATA_EXECUCAO}/dobra-{kfolds+1}_pesos_modelo.pth')     
        
    return metricasValidacao



PATH_DATA_EXECUCAO = f'./resultados/{datetime.now().strftime("%Y-%m-%d_%H-%M")}/'
RESULTADOS_CSV = f'{PATH_DATA_EXECUCAO}{RELATORIO_GERAL}'

verdadeiros, falsos = 3387, 3387
df_bert = Utils.DatasetSample.ObterDataset(dfPtBr, 378, 122) # oficial 3387, 3387
IniciarProcesso(df_bert, 'Português Balanceado (i)', f'lr-{LEARNING_RATE}_df-v{verdadeiros}-f{falsos}')