import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

_patch = f'../resultadoScripts/sentenceClassificationPre/'
_maxLength = 256
_model = 'bert-base-multilingual-uncased'
_modelSave = f'{_patch}model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(_model).to(device)
model.load_state_dict(torch.load(_modelSave))
tokenizer = BertTokenizerFast.from_pretrained(_model)

test_text = ["Grande variedade de shots de tequillas e mezcais para começar a noite no bairro, o bar abre cedo para depois ir jantar para 11Tapas , sugestão do sr. Manel ....", 
             "Pareces que á pessoas que avaliam sem ter estado nos locais!!! A Tasquinha de Baco é só um dos locais mais típicos em estilo moderno que existe em Lisboa! O atendimento é óptimo e a forma como acarinham os clientes é de louvar. Se passarem em Alfama não deixem de lá passar, comam um chourisso assado acompanhado de um belo Tinto! Vão sentir-se em casa!",
             "Jantar com um grupo de amigos. A expetativa era alta, pois alguém dizia que era o melhor restaurante italiano de Lisboa. Não fiquei impressionada. Restaurante pequeno, pouca privacidade por as mesas estarem demasiado juntas. Para 5 pessoas deram-nos uma mesa para quatro, com um no topo, que ficou encaixado entre um pilar e uma balustrada. Pedi um risotto de frutos do mar, que demonstrou estar passado do ponto, com frutos do mar de segunda ou terceira categoria. Staff atento e simpático. Estacionamento simplesmente impossível."]

tokens_test = tokenizer.batch_encode_plus(
    test_text,
    add_special_tokens=True,
    max_length=_maxLength,
    pad_to_max_length=True,
    return_token_type_ids=False,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
    truncation=True
)

input_ids = tokens_test['input_ids'].to(device)
attention_mask = tokens_test['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=1)

print(predictions) 

labels = ['Real', 'Fake']
predicted_labels = [labels[p] for p in predictions]

print(predicted_labels)