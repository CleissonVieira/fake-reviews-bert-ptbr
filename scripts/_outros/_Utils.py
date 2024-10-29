import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import torch
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup

class DatasetBert(Dataset):
    def __init__(self, encodings): self.encodings = encodings
    def __getitem__(self, idx): return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self): return len(self.encodings['input_ids'])

class Utils(Enum):
    DatasetCompleto, DatasetSample = 1, 2
    
    def ObterDataset(self, dfOriginal, quantidadeReais = 0, quantidadeFalsos = 0):
        dfReal, dfFake = dfOriginal[dfOriginal.fake_review == False], dfOriginal[dfOriginal.fake_review == True]
            
        if self == Utils.DatasetCompleto: df = pd.concat([dfReal, dfFake])
        elif self == Utils.DatasetSample: 
            df = pd.concat([dfReal.sample(n=quantidadeReais, random_state=42), dfFake.sample(n=quantidadeFalsos, random_state=42)])
        
        df['fake_review'] = df['fake_review'].astype(int)
        return df
    
    def ConfigurarOtimizadorPerdaAgendador(model, train_loader, learningRate, epochs):
        optimizer = AdamW(model.parameters(), lr=learningRate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(total_steps//10), num_training_steps=total_steps)
        return optimizer, scheduler
    
    def TokenizarTextos(texts, labels, tokenizer, maxLength):
        assert len(texts) == len(labels), "Número de textos e labels não correspondem!"
        inputs = tokenizer.batch_encode_plus(texts.tolist(), max_length=maxLength, padding='max_length', truncation=True, return_tensors="pt")
        inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        return inputs
    
    def ObterDataLoader(treino80, teste20, conteudo, booleanoRealFake, batchSize, tokenizer, maxLength):
        conteudoTreino60, conteudoTeste20 = conteudo[treino80], conteudo[teste20]
        realFakeTreino60, realFakeTeste20 = booleanoRealFake[treino80], booleanoRealFake[teste20]
        
        conteudoTreino60, conteudoValidacao20, realFakeTreino60, realFakeValidacao20 = train_test_split(
            conteudoTreino60, realFakeTreino60, test_size=0.25, stratify=realFakeTreino60, random_state=42
        )
        
        encodingsTreino = Utils.TokenizarTextos(conteudoTreino60, realFakeTreino60, tokenizer, maxLength)
        encodingsValidacao = Utils.TokenizarTextos(conteudoValidacao20, realFakeValidacao20, tokenizer, maxLength)
        encodingsTeste = Utils.TokenizarTextos(conteudoTeste20, realFakeTeste20, tokenizer, maxLength)
            
        dataLoaderTreino = DataLoader(DatasetBert(encodingsTreino), batch_size=batchSize, shuffle=True)
        dataLoaderValidacao = DataLoader(DatasetBert(encodingsValidacao), batch_size=batchSize)
        dataLoaderTeste = DataLoader(DatasetBert(encodingsTeste), batch_size=batchSize)
        
        return dataLoaderTreino, dataLoaderValidacao, dataLoaderTeste
    
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def FrequenciasPorTamanhoConteudoAvaliacao(dfPtBr, tokenizer): 
        dfMaxLength = dfPtBr
        dfMaxLength['review_length'] = dfMaxLength['content'].apply(len)
        dfMaxLength['num_tokens'] = dfMaxLength['content'].apply(lambda x: len(tokenizer.tokenize(x)))
        
        plt.hist(dfMaxLength['num_tokens'], bins=50)
        plt.xlabel('Número de Tokens')
        plt.ylabel('Frequência')
        plt.show()
        
    def GerarMatrizConfusao(y_true, y_pred, class_names, caminhoSalvar, nomeDf, nomePng):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.savefig(f'{caminhoSalvar}{nomeDf}/{nomePng}.png', format='png')
        
    def SalvarCsv(pathCsv, nomeCsv, resultados):
        patch = f'{pathCsv}{nomeCsv}'
        if not os.path.exists(f'{pathCsv}'): os.makedirs(f'{pathCsv}')
        if os.path.isfile(f'{patch}'): resultados.to_csv(f'{patch}', mode='a', header=False, index=False)
        else: resultados.to_csv(f'{patch}', mode='w', index=False)