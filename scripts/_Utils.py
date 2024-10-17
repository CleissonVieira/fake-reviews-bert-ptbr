import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
from enum import Enum
from sklearn.metrics import confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup

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
        optimizer = AdamW(model.parameters(), lr=learningRate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        return optimizer, scheduler
    
    def TokenizarTextos(texts, labels, tokenizer, maxLength):
        inputs = tokenizer(texts.tolist(), max_length=maxLength, padding=True, truncation=True, return_tensors="pt")
        inputs['labels'] = torch.tensor(labels)
        return inputs
    
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