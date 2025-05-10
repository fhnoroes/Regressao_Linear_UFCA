from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
import statsmodels.api as sm
import numpy as np


n = [10,20,40,80,100,500,1000]
distE = []
beta0 = []
beta1 = []

for i in n:
    #cria vetores de amostras com n crescendo 
    x = np.random.normal(0,1,i)
    y =  np.random.normal(0,0.5,i)
    #pega dist euclidiana com relação aos vetores x e y
    distE.append(euclidean(x,y))
    #add coluna de 1s ao vetor x
    novox = sm.add_constant(x)
    modelo = sm.OLS(y,novox).fit()
    #pega b0 e b1
    beta0.append(modelo.params[0])
    beta1.append(modelo.params[1])

#gráfico distancia euclidiana por n
plt.figure(figsize=(8,6))
plt.title("Gráfico distância Euclidiana por N")
plt.xlabel("valores de N")
plt.ylabel("valores da distância euclidiana")
plt.plot(n,distE, marker='o', color='green')
plt.grid(True)
plt.show()

#gráfico b0 por n
plt.figure(figsize=(8,6))
plt.title("Gráfio do Beta0 por N")
plt.xlabel("Valores de N")
plt.ylabel("Valores de B0")
plt.plot(n,beta0,marker='o',color='purple')
plt.grid(True)
plt.show()

#gráfico b1 por n
plt.figure(figsize=(8,6))
plt.title("Gráfico do beta1 por N")
plt.xlabel("Valores de N")
plt.ylabel("Valores de B1")
plt.plot(n,beta1, marker='o', color='blue')
plt.grid(True)
plt.show()