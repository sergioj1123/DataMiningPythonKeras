#foi utilizado o dataset fornecido pelo site https://forum.ailab.unb.br/t/datasets-em-portugues/251/7 disponivel no github https://github.com/americanas-tech/b2w-reviews01
#A ideia desse programa é predizer se o consumidor irá recomendar o produto que comprou a um amigo, servindo como base o texto que escreveu no titulo do comentario da compra.
#O modelo LSTM foi utilizado, porque as opções destinos seriam ou o consumidor recomenda ou não recomenda, sendo o formato ideal para o LSTM.
import pandas as pd
import nltk
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow import keras


#Eu optei por baixar o csv, para evitar problemas de internet.
avaliacoes = pd.read_csv('./avaliacoes.csv')

#reduzi o tamanho para não pesar tanto na hora de rodar
avaliacoes = avaliacoes[0:50000]


#excluindo os campos vazios da recomendação
avaliacoes.dropna(subset=['recommend_to_a_friend'])
#Aqui criei um dataset igual só para comparar antes e depois do pré processamento.
avaliacoes2 = avaliacoes


#Quantidade de pessoas que recomendariam para amigos ou nao recomendariam
print("Quantidade de pessoas que não recomendariam e que recomendariam da população total utilizada")
print((avaliacoes.recommend_to_a_friend == 'No').sum())
print((avaliacoes.recommend_to_a_friend == 'Yes').sum())

#Pré-processamento dos dados

#Função que transformar todos os texto resposta em binarios, para serem aceitos pelo LSTM
def converterEmBinario(texto):
    if(texto == 'No'):
        return '0'
    if(texto == 'Yes'):
        return '1'
    else:
        return '0'

#Função para remover caixa alta
def Lower(texto):
    novoTexto = str(texto)
    return novoTexto.lower()

#Função de remover stopwords, nessa etapa tive que remover a palavra "não" do conjunto de stop words, pois muitos clientes falavam que: "não recebi o produto", não recomendando para amigos.
def RemoveStopWords(texto):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    stopwords.remove('não')
    palavrasFiltradas = [word.lower() for word in texto.split() if word.lower() not in stopwords]
    return " ".join(palavrasFiltradas)

#Função para remover pontuação
def RemoverPontuacao(texto):
    translator = str.maketrans('', '', string.punctuation)
    return texto.translate(translator)

#Aplicando as funções do pré-processamento do dataset 

avaliacoes["review_title"] = avaliacoes.review_title.map(Lower)
avaliacoes["review_title"] = avaliacoes.review_title.map(RemoveStopWords)
avaliacoes["review_title"] = avaliacoes.review_title.map(RemoverPontuacao)
avaliacoes["recommend_to_a_friend"] = avaliacoes.recommend_to_a_friend.map(converterEmBinario)



#Função que pega conta a quantidade de vezes que cada palavra aparece

def ContadorDeRepeticoes(coluna):
    count = Counter()
    for texto in coluna.values:
        for word in texto.split():
            count[word] += 1
    return count

numeroPalavras = ContadorDeRepeticoes(avaliacoes.review_title)

#Inciando a divisão de treinamento e teste, defindo treino como 80% e teste como 20%

tamanho_treino = int(avaliacoes.shape[0]*0.8)

vetor_treino = avaliacoes[:tamanho_treino]
vetor_teste = avaliacoes[tamanho_treino:]

#Defindo oque será usado como alvo e como base do treinamento
treino_sentenca = vetor_treino.review_title.to_numpy()
treino_label = vetor_treino.recommend_to_a_friend.to_numpy()
teste_sentenca = vetor_teste.review_title.to_numpy()
teste_label = vetor_teste.recommend_to_a_friend.to_numpy()

#convertendo os labels para um formato aceito pela biblioteca
treino_label = np.asarray(treino_label).astype("float64")
teste_label = np.asarray(teste_label).astype("float64")



#Iniciando o processo de tokenlize

tokenizer = Tokenizer(len(numeroPalavras))
tokenizer.fit_on_texts(treino_sentenca)

#indexando cada palavra unica
word_index = tokenizer.word_index

#Tokenlize sentença de treino e teste

sequenciaTreino = tokenizer.texts_to_sequences(treino_sentenca)
sequenciaTeste = tokenizer.texts_to_sequences(teste_sentenca)

#Tamanho maximo das palavras
max_length = 25

treino_padded = pad_sequences(sequenciaTreino, maxlen = max_length, padding = 'post', truncating = 'post')
teste_padded = pad_sequences(sequenciaTeste, maxlen = max_length, padding = 'post', truncating = 'post')

#função executada para fazer o caminho contrario da tokenlize
reverso_index = dict([(idx, word) for (word, idx) in word_index.items()])

def decodificar(sequencia):
    return " ".join([reverso_index.get(idx, "?") for idx in sequencia])


#Validando se tanto a tokenizer quanto o caminho reverso apontam para a mesma palavra
print("Validando se tanto a tokenizer quanto o caminho reverso apontam para a mesma palavra")
print(decodificar(sequenciaTreino[10]))
print(sequenciaTreino[10])
print(treino_sentenca[10])

#Iniciando o treinamento usando o modelo LSTM

modelo = keras.models.Sequential()
modelo.add(layers.Embedding((len(numeroPalavras)), 32, input_length = max_length))
modelo.add(layers.LSTM(64, dropout = 0.1))
modelo.add(layers.Dense(1, activation = "sigmoid"))
modelo.summary()

loss = keras.losses.BinaryCrossentropy(from_logits = False)
optim = keras.optimizers.Adam(lr = 0.001)
metrics = ["accuracy"]

modelo.compile(loss=loss, optimizer = optim, metrics=metrics)

#define o número de pepocas em 20 para evitar que acabe demorando muito
modelo.fit(treino_padded, treino_label, epochs=20, validation_data=(teste_padded, teste_label), verbose=2)

predicao = modelo.predict(treino_padded)
predicao = [1 if p > 0.5 else 0 for p in predicao]


#Exemplo final só para verificar como o algoritmo classificou
print(treino_sentenca[10:20])
print(treino_label[10:20])
print(predicao[10:20])




