
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

_patch = f'../resultadoScripts/sentenceClassPreTrain/'
_maxLength = 512
_model = 'bert-base-multilingual-uncased'
_modelSave = f'{_patch}model.pt'

print("Recuperando modelo para testar")

class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomBertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(_model)
        self.pre_classifier = nn.Linear(768, 768)  
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output[0]  
        pooled_output = hidden_state[:, 0]  
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output) 
        logits = self.classifier(pooled_output)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomBertForSequenceClassification().to(device)
model.load_state_dict(torch.load(_modelSave))
tokenizer = BertTokenizerFast.from_pretrained(_model)

def predict_sentiment(review_text, model, tokenizer, max_length = _maxLength):
    model.eval()

    encoding = tokenizer.encode_plus(
          review_text,
          add_special_tokens=True,
          max_length=max_length,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    prediction = torch.argmax(logits, dim=1).item()

    label_dict = {0: 'Real', 1: 'Fake'}
    sentiment = label_dict[prediction]

    return sentiment


review_1 = "Grande variedade de shots de tequillas e mezcais para começar a noite no bairro, o bar abre cedo para depois ir jantar para 11Tapas , sugestão do sr. Manel ...."
review_2 = "Pareces que á pessoas que avaliam sem ter estado nos locais!!! A Tasquinha de Baco é só um dos locais mais típicos em estilo moderno que existe em Lisboa! O atendimento é óptimo e a forma como acarinham os clientes é de louvar. Se passarem em Alfama não deixem de lá passar, comam um chourisso assado acompanhado de um belo Tinto! Vão sentir-se em casa!"
review_3 = "Jantar com um grupo de amigos. A expetativa era alta, pois alguém dizia que era o melhor restaurante italiano de Lisboa. Não fiquei impressionada. Restaurante pequeno, pouca privacidade por as mesas estarem demasiado juntas. Para 5 pessoas deram-nos uma mesa para quatro, com um no topo, que ficou encaixado entre um pilar e uma balustrada. Pedi um risotto de frutos do mar, que demonstrou estar passado do ponto, com frutos do mar de segunda ou terceira categoria. Staff atento e simpático. Estacionamento simplesmente impossível."
review_4 = "Sei la o que é isso"
review_5 = "Excelente opção no Baixo Chiado. Provamos a portuguesinha, uma espécie de empada de cozido à portuguesa e polvo com alho em molho kimchi com batata doce. Muito bom. Além disso o queijo serra da estrela amanteigado com mais o presunto bísaro (porco preto defumado) também foram demais. Definitivamente uma ótima opção no circuito gastronômico de Lisboa."

print("Fake = ",predict_sentiment(review_1, model, tokenizer))
print("Fake = ",predict_sentiment(review_2, model, tokenizer))
print("Real = ",predict_sentiment(review_3, model, tokenizer))
print("Fake = ",predict_sentiment(review_4, model, tokenizer))
print("Fake = ",predict_sentiment(review_5, model, tokenizer))