import numpy as np
import os
import pandas as pd
import time
import torch
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from scripts._outros._Utils import Utils, DatasetBert

warnings.filterwarnings('ignore')
Utils.set_seed(42)

urlPtBr = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
dfPtBr = pd.read_csv(urlPtBr)
dfPtBr = dfPtBr[['content', 'fake_review']]

RELATORIO_GERAL = 'RelatorioTreinoValidacao.csv'
TOKENIZADOR = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 5e-5

_resultadosDf = []

def TreinarModelo(modelo, train_loader, otimizador, agendador):
    perdaTotal = 0

    for batch in train_loader:
        otimizador.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        saidas = modelo(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=batch['labels']) 
        
        perda = saidas.loss
        perdaTotal += perda.item()
        perda.backward()
        otimizador.step()
        agendador.step()

    return perdaTotal / len(train_loader)

def AvaliarTestarModelo(modelo, valLoader, nomeDf, nomePng):
    preds, trueLabels = [], []
    modelo.eval()

    with torch.no_grad():
        for batch in valLoader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            saidas = modelo(**batch)
            
            saidas = modelo(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=batch.get('labels'))
            
            logits = saidas.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            trueLabels.extend(batch['labels'].cpu().numpy())

    report = classification_report(trueLabels, preds, target_names=['Real', 'Fake'], output_dict=True)
    dfClassificationReport = pd.DataFrame(report).transpose()
    dfTitulo = pd.DataFrame([[f'Classification Report ({nomeDf} - {datetime.now().strftime("%Y-%m-%d_%H-%M")}):']])
    file_exists = os.path.isfile(f'{PATH_DATA_EXECUCAO}relatorioMatrizConfusao.csv')
    
    with open(f'{PATH_DATA_EXECUCAO}relatorioMatrizConfusao.csv', 'a') as f:
        dfTitulo.to_csv(f, index=False, header=not file_exists)
        dfClassificationReport.to_csv(f, index=True, header=True) 
        f.write('\n')
    
    # Utils.GerarMatrizConfusao(trueLabels, preds, ['Real', 'Fake'], PATH_DATA_EXECUCAO, nomeDf, nomePng)

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
    temporizador = time.time()
    
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
    
    treino80, teste20, treino80_realFake, teste20_realFake = train_test_split(
        dfUtilizado['content'].values, dfUtilizado['fake_review'].values, 
        test_size=0.2, 
        stratify=dfUtilizado['fake_review'].values,  # Mantém a proporção entre as classes
        random_state=42 
    )
    
    conteudoTreino60, conteudoValidacao20, realFakeTreino60, realFakeValidacao20 = train_test_split(
        treino80, treino80_realFake, 
        test_size=0.20,  # 0.20 de 80% -> 16% do total
        stratify=treino80_realFake, 
        random_state=42
    )
    
    encodingsTreino = Utils.TokenizarTextos(conteudoTreino60, realFakeTreino60, TOKENIZADOR, MAX_LENGTH)
    encodingsValidacao = Utils.TokenizarTextos(conteudoValidacao20, realFakeValidacao20, TOKENIZADOR, MAX_LENGTH)
    encodingsTeste = Utils.TokenizarTextos(teste20, teste20_realFake, TOKENIZADOR, MAX_LENGTH)

    # Criação dos DataLoaders
    dataLoaderTreino = DataLoader(DatasetBert(encodingsTreino), batch_size=BATCH_SIZE, shuffle=True)
    dataLoaderValidacao = DataLoader(DatasetBert(encodingsValidacao), batch_size=BATCH_SIZE)
    dataLoaderTeste = DataLoader(DatasetBert(encodingsTeste), batch_size=BATCH_SIZE)
    
    modeloUtilizado = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
    modeloUtilizado.to(DEVICE)
    otimizador, agendador = Utils.ConfigurarOtimizadorPerdaAgendador(modeloUtilizado, dataLoaderTreino, LEARNING_RATE, EPOCHS)

    try:
        for ep in range(EPOCHS):
            modeloUtilizado.train()
            temporizador = time.time()
            perdaTreino = TreinarModelo(modeloUtilizado, dataLoaderTreino, otimizador, agendador)
            print(f"Treino: Epoch {ep+1} | loss {perdaTreino:.4f} Memória GPU alocada: {torch.cuda.memory_allocated()/1024/1024} MB")
            temporizadorTreino = time.time() - temporizador

            nomePng = f'Val_Epoch-{ep+1}'
            temporizador = time.time()
            precision, recall, f1, precisionReal, precisionFake, recallReal, recallFake, f1Real, f1Fake = AvaliarTestarModelo(modeloUtilizado, dataLoaderValidacao, nomeDf, nomePng)
            temporizadorValidacao = time.time() - temporizador
            print(f"Validação: Epoch {ep+1} | F1-Score {f1:.4f}, Precision {precision:.4f}, Recall {recall:.4f}")
            
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
            
    except KeyboardInterrupt:
        torch.save(modeloUtilizado.state_dict(), f'{PATH_DATA_EXECUCAO}/pesos_modelo.pth')
        print("\nTreinamento interrompido. Indo diretamente para o teste.")
        
    # entender se é para substituir as métricas adicionadas da validação pelas de teste
    # se eu retorno e salvo o modelo treinado, salvo do último kfold ou misturo os resultados para formar o modelo
    
    print(f"Avaliação no conjunto de teste")
    nomePng = f'Test_Epoch-{EPOCHS}'
    precision_teste, recall_teste, f1_teste, precisionReal_teste, precisionFake_teste, recallReal_teste, recallFake_teste, f1Real_teste, f1Fake_teste = AvaliarTestarModelo(modeloUtilizado, dataLoaderTeste, nomeDf, nomePng)
    
    print(f"Teste: F1-Score {f1_teste:.4f}, Precision {precision_teste:.4f}, Recall {recall_teste:.4f}")
    
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
        'epoch': EPOCHS,
        'batch_size': BATCH_SIZE,
        'max_length': MAX_LENGTH,
        'learning_rate': LEARNING_RATE,
        'val_f1_score': np.mean(metricasValidacao['F1_Score']),
        'test_f1_score': np.mean(metricasTeste['F1_Score']),
    }]).round(5)

    if os.path.isfile(RESULTADOS_CSV): resultadosDf.to_csv(RESULTADOS_CSV, mode='a', header=False, index=False)
    else: resultadosDf.to_csv(RESULTADOS_CSV, mode='w', index=False)
    
    _resultadosDf.append(resultadosDf)
    return metricasValidacao



PATH_DATA_EXECUCAO = f'../resultado/_epoch-{datetime.now().strftime("%Y-%m-%d_%H-%M")}/'
RESULTADOS_CSV = f'{PATH_DATA_EXECUCAO}{RELATORIO_GERAL}'

verdadeiros, falsos = 3387, 3387 
df_bert = Utils.DatasetSample.ObterDataset(dfPtBr, verdadeiros, falsos) 
IniciarProcesso(df_bert, 'Português Balanceado (i)', f'lr-{LEARNING_RATE}_df-v{verdadeiros}-f{falsos}')

print(_resultadosDf)