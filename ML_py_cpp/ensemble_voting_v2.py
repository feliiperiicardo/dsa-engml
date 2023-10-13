# Modelagem de Tópico de Noticiário Financeiro

#Imports
import numpy as np
#from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Carregando dados
noticias = load_files('/Users/felipefernandes/Desktop/DSA/dsa-engml/ML_py_cpp/dados/bbc',
                      encoding= 'utf-8', decode_error= 'replace')

X = noticias.data
y = noticias.target

# Para gravar os resultados
resultado = {}
resultado['hard'] = []
resultado['soft'] = []

for seed in range(1,50,10):

    # Divisão em treino e teste (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state= seed)

    # Vetorização
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    vectorizer = TfidfVectorizer(norm=None, stop_words='english', max_features=1000, decode_error='ignore')

    # Aplicamos a vetorização
    # Observe que treinamos e aplicamos em treino e apenas aplicamos em teste 
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)


    # Criando 3 modelos diferentes

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=30, max_iter=1000)

    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=1)

    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    nb_clf = MultinomialNB()

    # Iniciando o modelo de votação
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    for v in ['hard', 'soft']:

        voting_model = VotingClassifier(estimators=[('lr', lr_clf), ('rf', rf_clf),('nb', nb_clf)], voting=v)
        # Treinamento
        voting_model = voting_model.fit(X_train_vectors, y_train)

        # Previsões com dados de teste
        previsoes = voting_model.predict(X_test_vectors) 

        print(f'-Random State: {seed}, -Voting: {v}, -Acurácia: {accuracy_score(y_test, previsoes)}')       
        resultado[v].append((seed, accuracy_score(y_test, previsoes)))

print('\nMelhores Resultados: ')

# Extrai os melhores resultados
h = max(resultado['hard'], key=lambda x: x[1])
s = max(resultado['soft'], key=lambda x: x[1])

#print('h: ', h)
#print('s: ', s)

# Print
print(f'\n-Random State: {h[0]}, -Voting: hard, -Acurácia: {h[1]}')
print(f'\n-Random State: {s[0]}, -Voting: soft, -Acurácia: {s[1]}')




