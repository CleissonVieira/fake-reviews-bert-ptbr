import numpy as np
import os
import pandas as pd
import time
import torch
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from scripts._outros._Utils import Utils

warnings.filterwarnings('ignore')
Utils.set_seed(42)

urlPtBr = 'https://raw.githubusercontent.com/CleissonVieira/fake-reviews-bert-ptbr/main/datasets/yelp-fake-reviews-dataset-pt.csv'
dfPtBr = pd.read_csv(urlPtBr)
dfPtBr = dfPtBr[['content', 'fake_review']]

RELATORIO_GERAL = 'RelatorioTreinoValidacao.csv'
TOKENIZADOR = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

KFOLDS = 2
EPOCHS = 20
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 3e-5

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
    perdaTotal = 0
    modelo.eval()

    with torch.no_grad():
        for batch in valLoader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            saidas = modelo(**batch)
            
            saidas = modelo(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=batch.get('labels'))
            
            logits = saidas.logits
            perdaTotal += saidas.loss.item()
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

    return perdaTotal, precision, recall, f1, precision_real, precision_fake, recall_real, recall_fake, f1_real, f1_fake


    
    
# MÉTODO PRINCIPAL

def IniciarProcesso(dfUtilizado, conjuntoDados, nomeDf):
    print(f'\nLR {LEARNING_RATE} | EP {EPOCHS} | Dataset {conjuntoDados} | Verdadeiros: {len(dfUtilizado[dfUtilizado.fake_review == False])} | Falsos: {len(dfUtilizado[dfUtilizado.fake_review == True])}')
    if not os.path.exists(f'{PATH_DATA_EXECUCAO}{nomeDf}/'): os.makedirs(f'{PATH_DATA_EXECUCAO}{nomeDf}/')

    temporizador = time.time()
    skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=42) 
    
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
    
    conteudoAvaliacao, booleanoRealFake = dfUtilizado['content'].values, dfUtilizado['fake_review'].values
    
    for folds, (treino80, teste20) in enumerate(skf.split(conteudoAvaliacao, booleanoRealFake)):
        dataLoaderTreino, dataLoaderValidacao, dataLoaderTeste = Utils.ObterDataLoader(treino80, teste20, conteudoAvaliacao, booleanoRealFake, BATCH_SIZE, TOKENIZADOR, MAX_LENGTH)
        modeloUtilizado = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
        modeloUtilizado.to(DEVICE)
        otimizador, agendador = Utils.ConfigurarOtimizadorPerdaAgendador(modeloUtilizado, dataLoaderTreino, LEARNING_RATE, EPOCHS)
    
        for ep in range(0, EPOCHS, 5):
            modeloUtilizado.train()
            temporizador = time.time()
            for epoch in range(5):
                perdaTreino = TreinarModelo(modeloUtilizado, dataLoaderTreino, otimizador, agendador)
                print(f"Treino: fold {folds + 1} | Epoch {ep + 1} até {ep + 5} ({ep + 1 + epoch}) | loss {perdaTreino:.4f} Memória GPU alocada: {torch.cuda.memory_allocated()/1024/1024} MB")
            temporizadorTreino = time.time() - temporizador

            nomePng = f'Val_Epoch-{ep+1}a{ep+5}_Dobra-{folds+1}'
            temporizador = time.time()
            perda, precision, recall, f1, precisionReal, precisionFake, recallReal, recallFake, f1Real, f1Fake = AvaliarTestarModelo(modeloUtilizado, dataLoaderValidacao, nomeDf, nomePng)
            temporizadorValidacao = time.time() - temporizador
            print(f"Avaliação do modelo:  perda {perda:.4f} | Epoch {ep + 1} até {ep + 5}")
        
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
            
            print(f"Validação: F1-Score {f1:.4f}, Precision {precision:.4f}, Recall {recall:.4f}")
        
        nomePng = f'Test_Epoch-{EPOCHS}_Dobra-{folds+1}'
        perda, precision_teste, recall_teste, f1_teste, precisionReal_teste, precisionFake_teste, recallReal_teste, recallFake_teste, f1Real_teste, f1Fake_teste = AvaliarTestarModelo(modeloUtilizado, dataLoaderTeste, nomeDf, nomePng)
        
        print(f"Teste: F1-Score {f1_teste:.4f}, Precision {precision_teste:.4f}, Recall {recall_teste:.4f}")
        print(f"Teste do modelo: perda {perda:.4f} | Epoch {ep + 1} até {ep + 5}")
        
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
            'fold_epoch': f'{folds + 1} | {EPOCHS}',
            'batch_size': BATCH_SIZE,
            'max_length': MAX_LENGTH,
            'learning_rate': LEARNING_RATE,
            # 'tempo_treino_segundos': metricasValidacao['tempoTreino'],
            # 'tempo_validacao_segundos': metricasValidacao['tempoValidacao'],
            # 'avaliacoes_dataLoader': f'{len(dfUtilizado)} | v {len(dfUtilizado[dfUtilizado.fake_review == False])} | f {len(dfUtilizado[dfUtilizado.fake_review == True])}',
            # 'dataset': conjuntoDados,
            'val_f1_score': np.mean(metricasValidacao['F1_Score']),
            'test_f1_score': np.mean(metricasTeste['F1_Score']),
            # 'val_f1_score_real': np.mean(metricasValidacao['F1_Score_Real']),
            # 'test_f1_score_real': np.mean(metricasTeste['F1_Score_Real']),
            # 'val_f1_score_fake': np.mean(metricasValidacao['F1_Score_Fake']),
            # 'test_f1_score_fake': np.mean(metricasTeste['F1_Score_Fake']),
            # 'val_f1_score_variance': np.var(metricasValidacao['F1_Score'], ddof=1),
            # 'test_f1_score_variance': np.var(metricasTeste['F1_Score'], ddof=1),
            # 'val_f1_score_min': np.min(metricasValidacao['F1_Score']),
            # 'test_f1_score_min': np.min(metricasTeste['F1_Score']),
            # 'val_f1_score_max': np.max(metricasValidacao['F1_Score']),
            # 'test_f1_score_max': np.max(metricasTeste['F1_Score']),
            # 'val_precision_real': np.mean(metricasValidacao['Precision_Real']),
            # 'test_precision_real': np.mean(metricasTeste['Precision_Real']),
            # 'val_precision_fake': np.mean(metricasValidacao['Precision_Fake']),
            # 'test_precision_fake': np.mean(metricasTeste['Precision_Fake']),
            # 'val_precision': np.mean(metricasValidacao['Precision']),
            # 'test_precision': np.mean(metricasTeste['Precision']),
            # 'val_precision_variance': np.var(metricasValidacao['Precision'], ddof=1),
            # 'test_precision_variance': np.var(metricasTeste['Precision'], ddof=1),
            # 'val_precision_min': np.min(metricasValidacao['Precision']),
            # 'test_precision_min': np.min(metricasTeste['Precision']),
            # 'val_precision_max': np.max(metricasValidacao['Precision']),
            # 'test_precision_max': np.max(metricasTeste['Precision']),
            # 'val_recall_real': np.mean(metricasValidacao['Recall_Real']),
            # 'test_recall_real': np.mean(metricasTeste['Recall_Real']),
            # 'val_recall_fake': np.mean(metricasValidacao['Recall_Fake']),
            # 'test_recall_fake': np.mean(metricasTeste['Recall_Fake']),
            # 'val_recall': np.mean(metricasValidacao['Recall']),
            # 'test_recall': np.mean(metricasTeste['Recall']),
            # 'val_recall_variance': np.var(metricasValidacao['Recall'], ddof=1),
            # 'test_recall_variance': np.var(metricasTeste['Recall'], ddof=1),
            # 'val_recall_min': np.min(metricasValidacao['Recall']),
            # 'test_recall_min': np.min(metricasTeste['Recall']),
            # 'val_recall_max': np.max(metricasValidacao['Recall']),
            # 'test_recall_max': np.max(metricasTeste['Recall'])
        }]).round(5)

        if os.path.isfile(RESULTADOS_CSV): resultadosDf.to_csv(RESULTADOS_CSV, mode='a', header=False, index=False)
        else: resultadosDf.to_csv(RESULTADOS_CSV, mode='w', index=False)
        
        # torch.save(modeloUtilizado.state_dict(), f'{PATH_DATA_EXECUCAO}/dobra-{folds+1}_pesos_modelo.pth')     
        
    return metricasValidacao



PATH_DATA_EXECUCAO = f'../resultadoScripts/foldEpoch-{datetime.now().strftime("%Y-%m-%d_%H-%M")}/'
RESULTADOS_CSV = f'{PATH_DATA_EXECUCAO}{RELATORIO_GERAL}'

verdadeiros, falsos = 3387, 3387 # oficial 3387, 3387
df_bert = Utils.DatasetSample.ObterDataset(dfPtBr, verdadeiros, falsos) 
IniciarProcesso(df_bert, 'Português Balanceado (i)', f'lr-{LEARNING_RATE}_df-v{verdadeiros}-f{falsos}')