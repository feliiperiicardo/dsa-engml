# Projeto 2 - Prevendo o retorno financeiro de investimentos em títulos públicos

# Abordagem utilizando o framework

# Imports
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Carregando o Dataset
df = pd.read_csv('dados/dataset.csv')
print(f'\nDados Carregados com Sucesso!\nShape: {df.shape}')
print(df.head())


# Visualizando os Dados
df.plot(x = 'Investimento', y = 'Retorno', style='o')
plt.title('Investimento x Retorno')
plt.xlabel('Investimento')
plt.ylabel('Retorno')
plt.savefig('imagens/parte1-grafico1.png')
plt.show()

# Preparando os Dados
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Divisão em dados de treino e teste (70/30)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

X_treino = X_treino.reshape(-1, 1).astype(np.float32)

# Construção do Modelo

# Modelo de Regressão Linear
reg = LinearRegression()

# Treinamento do modelo
reg.fit(X_treino, y_treino)
print('\nModelo Treinado com Sucesso!')

# Imprimindo os coeficientes B0 e B1
print(f'\nB1 (coef_): {reg.coef_}')
print(f'B0 (intercept_): {reg.intercept_}')


# Plot da linha de regressão linear
# y = B0 + B1 * X
regression_line = reg.intercept_ + reg.coef_ * X 
plt.scatter(X, y)
plt.title('Investimento x Retorno')
plt.xlabel('Investimento')
plt.ylabel('Retorno Previsto')
plt.plot(X, regression_line, color='red')
plt.savefig('imagens/parte1-regressionLine.png')
plt.show()

# Previsões com dados de teste
y_pred = reg.predict(X_teste)

# Real x Previsto
df_valores = pd.DataFrame({'Valor Real': y_teste, 'Valor Previsto': y_pred})
print('\n')
print(df_valores)

# Plot
fig, ax = plt.subplots()
index = np.arange(len(X_teste))
bar_width = 0.35
actual = plt.bar(index, df_valores['Valor Real'], bar_width, label = 'Valor Real')
predicted = plt.bar(index + bar_width, df_valores['Valor Previsto'], bar_width, label = 'Valor Previsto')
plt.xlabel('Investimento')
plt.ylabel('Retorno Previsto')
plt.title('Valor Real x Valor Previsto')
plt.xticks(index + bar_width, X_teste)
plt.legend()
plt.savefig('imagens/parte1-actualvspredicted.png')
plt.show()

# Avaliação do modelo
print('\n')
print('MAE (Mean Absolute Error):', mean_absolute_error(y_teste, y_pred))
print('MSE (Mean Squared Error):', mean_squared_error(y_teste, y_pred))
print('RMSE (Root Mean Squared Error):', math.sqrt(mean_squared_error(y_teste, y_pred)))
print('R2 Score:', r2_score(y_teste, y_pred))

# Prevendo retorno para o investimento com novos dados (deploy)

# Recebendo input via terminal e aplicando aos dados as mesmas transformações aplicadas aos dados de treino
print('\n')
input_inv = float(input('\nDigite o valor que deseja investir: '))
inv = np.array([input_inv])
inv = inv.reshape(-1, 1)

# Previsões
pred_score = reg.predict(inv)

print('\n')
print(f'Investimento Realizado = {input_inv}')
print(f'Retorno Previsto = {pred_score[0]:.4}\n')