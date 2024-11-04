# Detecção de Avaliações Falsas em Português Utilizando Aprendizado Profundo

## Artigo
- Aguardando conclusão ...

## Sobre o Projeto
- Testado diversas configurações para o ajuste fino
- O projeto é uma continuação do estudo incluso no repositório [yelp-fake-reviews-ptbr](https://github.com/lucaspercisi/yelp-fake-reviews-ptbr). 
- O dataset utilizado é o mesmo para possibilitar a comparação entre os estudos.

## Objetivo
O objetivo principal é encontrar a melhor configuração no ajuste fino para um modelo de aprendizado profundo na identificação de avaliações falsas em português e comparar os resultados obtidos com os resultados do estudo [yelp-fake-reviews-ptbr](https://github.com/lucaspercisi/yelp-fake-reviews-ptbr).


## Contato
- **Cleisson Vieira** - cleisson.raimundi@gmail.com

## Termos técnicos
### Úteis e essencias para realizar o ajuste fino de acordo com a tarefa desejada
- Overfitting: o modelo aprende tão bem os detalhes e o ruído do conjunto de treino que perde a capacidade de generalizar para novos dados.
- Underfitting: o modelo não aprende o suficiente nem o padrão básico dos dados, então seu desempenho é ruim tanto nos dados de treino quanto nos de teste.
- DropOut: desativa neurônios durante o treino. Quanto maior o DropOut mais ajuda a evitar overfitting, porém dificulda o aprendizado. Quando menor, aprende mais rapido o que pode ocasionar overfitting *tf.keras.layers.Dropout(0.5)*.
- Weight Decay: adiciona penalidade a grandes pesos ajudando a regularizar e capturar bem os padrões de treino. Precisa encontrar um equilibrio. Muito alto ajuda a regulazir o modelo mas ocasiona overfitting. Muito menor não capturar bem os padrões de treino e causa underfitting
- Max Length: é o tamanho dos textos para o modelo processar, precisa ser definido de forma ideal, balanceando detalhes e concisão. Se for muito grande/longo, pode incluir muitos detalhes irrelevantes. Se for muito curto, pode faltar informações importantes.
- Batch size: é o número de amostras que o modelo processa de cada vez, equilibrando eficiência e precisão. Se muitos de uma vez (batch grande), pode não treinar uniformemente. Se poucos (batch pequeno), demora muito para treinar.
- No treinamento do modelo, epochs são quantas vezes o conjunto de dados de treino completo é passado pelo modelo. Muitas epochs podem levar ao overfitting, poucas podem não ser suficientes para aprender.
- Learning Rate: é o tamanho do ajuste que o modelo faz em seus parâmetros em cada iteração para reduzir o erro. Learning rate alto causa instabilidade e não consegue encontrar a solução ideal. Learning rate baixo aprende devagar e pode não alcançar a precisão ideal de maneira eficiente.
- Folds: Cross-validation usa várias subdivisões (folds) dos dados para testar e treinar o modelo, garantindo que ele generalize bem em diferentes partes dos dados.

---

Link para o Repositório: [https://github.com/CleissonVieira/fake-reviews-bert-ptbr](https://github.com/CleissonVieira/fake-reviews-bert-ptbr)




CONCLUSÕES
- O conjunto de dados é considerado pequeno, e por isso é fácil dar overfiting
- Com isso, o ideal é diminuir o Learning rate e as épocas (verificado em artigo sobre instabilidade no fine-tuning acima de 5 epocas)

CONVERSA COM PERCISI:
- o melhor cenário dele incluiu apenas features numéricas
- a feature textual (content) diminui o f1-score

O Percisi pegou os resultados do teste, coletado durante o cross_validate
No código dele, acrescentei o train_test_split, separei 20% para realizar o teste separado. O f1-score varia pouco entre validação e teste

# Tabela de execuções 
1. **ok** Executar KNN,Word2Vec 64_16_20 com feature content
2. **ok** Executar KNN,Word2Vec 64_16_20 com features numéricas
3. **ok** Executar KNN,Word2Vec 64_16_20 sem features content e numéricas

# Weight_decay de 0.03, early_stopping_patience de 3, sem DropOut
4. **ok** Executar BERT,BERT 64_16_20 feature content
5. **ok** Executar BERT,BERT 64_16_20 features numéricas sem legenda
6. **ok** Executar BERT,BERT 64_16_20 features numéricas com legenda
7. **ok** Executar BERT,BERT 64_16_20 features content e numéricas sem legenda
8. **ok** Executar BERT,BERT 64_16_20 features content e numéricas com legenda

# Weight_decay 0.01, early_stopping_patience 2, com DropOut 0.5
4. **ok** Executar BERT,BERT 64_16_20 feature content
5. **ok** Executar BERT,BERT 64_16_20 features numéricas sem legenda
6. **ok** Executar BERT,BERT 64_16_20 features numéricas com legenda
7. **ok** Executar BERT,BERT 64_16_20 features content e numéricas sem legenda
8. **ok** Executar BERT,BERT 64_16_20 features content e numéricas com legenda

# Retestar cenários abaixo
# Utilizar: LR 1e-3, Epochs 20, batch_size 32, ealy_stopping 5
# Com as alterações e apenas features numéricas (código separado)
5. **ok** Executar BERT,BERT 64_16_20 features numéricas sem legenda

# Weight_decay 0.01, early_stopping_patience 2, com DropOut 0.5
5. **ok** Executar BERT,BERT 64_16_20 features numéricas sem legenda

# O melhor de todos os cenários acima testar com LR 1e-05 e 5e-05
# O melhor entre os LRs 1e-05, 3e-05 e 5e-05 testar com 10 épocas ou mais