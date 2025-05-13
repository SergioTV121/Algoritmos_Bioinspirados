#Tinoco Videgaray Sergio Ernesto 

#Mochila
import numpy as np

v=(2,3,5,5,6) #Tupla v
w=(2,4,4,5,7) #Tupla w
W=9
N=len(v)  #5

m=np.zeros((N+1,W+1)) #Matriz de 0's de 6x10

for row in range(1,N+1):
  for col in range(1,W+1):
    if w[row-1]>col:
      m[row][col]=m[row-1][col]
    else:
      m[row][col]=max(m[row-1][col],m[row-1][col-w[row-1]]+v[row-1])

m


#Monedas
v=(1,2,5) #Tupla d
N=7
W=len(v)  #3

m=np.zeros((W,N+1)) #Matriz de 0's de 3x8

for i in range(N+1):
    m[0][i]=i


for row in range(1,W):
  for col in range(1,N+1):
    if v[row]>col:
      m[row][col]=m[row-1][col]
    else:
      m[row][col]=min(m[row-1][col],m[row][col-v[row]]+1)

m