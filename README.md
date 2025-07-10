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
