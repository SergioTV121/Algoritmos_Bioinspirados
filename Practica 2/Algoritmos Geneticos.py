#Tinoco Videgaray Sergio Ernesto
#Algoritmos Bioinspirados 5BV1


import ipywidgets as widgets
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython import display as display

from functools import cmp_to_key

import openturns as ot
import openturns.viewer as viewer
import numpy as np

ot.Log.Show(ot.Log.NONE)


#Funcion de Rastringin
def f(X):
    A = 10.0
    delta = [x**2 - A * np.cos(2 * np.pi * x) for x in X]
    y = A + sum(delta)
    return [y]

rastrigin = ot.PythonFunction(2, 1, f)  
rastrigin = ot.MemoizeFunction(rastrigin)
 


def create_button():
  button = widgets.Button(
    description='Next Generation',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Next Generation',
    icon='check' # (FontAwesome names without the `fa-` prefix)
  )
  return button




#Chromosomes are 4 bits long
L_chromosome=4
N_chains=2**L_chromosome
#Lower and upper limits of search space
a=-5
b=5
crossover_point=int(L_chromosome/2)


def random_chromosome():  #Genera un cromosoma con 4 bits
    chromosome=[]
    for i in range(0,L_chromosome):
        if random.random()<0.1:
            chromosome.append(0)
        else:
            chromosome.append(1)

    return chromosome

#Number of chromosomes
N_chromosomes=10
#probability of mutation
prob_m=0.15

F0=[]   #Vector de cromosomas.
fitness_values=[]   #Vector de ajuste.

for i in range(0,N_chromosomes):    #Genera una cadena de 10 cromosomas 
    F0.append(random_chromosome())
    fitness_values.append(0)


#binary codification
def decode_chromosome(chromosome):
    global L_chromosome,N_chains,a,b
    value=0
    for p in range(L_chromosome):
        value+=(2**p)*chromosome[-1-p]

    return ((b-a)*float(value)/(N_chains-1))+a



def evaluate_chromosomes():
    global F0

    for p in range(N_chromosomes):
        v1=decode_chromosome(F0[p])
        fitness_values[p]=f([v1])
        

def compare_chromosomes(chromosome1,chromosome2):
    vc1=decode_chromosome(chromosome1)
    vc2=decode_chromosome(chromosome2)
    
    fvc1=f([vc1])
    fvc2=f([vc2])
    
    if fvc1 > fvc2:
        return 1
    elif fvc1 == fvc2:
        return 0
    else: #fvg1<fvg2
        return -1
    

Lwheel=N_chromosomes*10

def create_wheel():
    global F0,fitness_values
   
    maxv=max(fitness_values)
    acc=0
    for p in range(N_chromosomes):
       acc+=maxv-fitness_values[p][0]
    fraction=[]
    for p in range(N_chromosomes):
        fraction.append( float(maxv-fitness_values[p][0])/acc)
        if fraction[-1]<=1.0/Lwheel:
            fraction[-1]=1.0/Lwheel
##    print fraction
    fraction[0]-=(sum(fraction)-1.0)/2
    fraction[1]-=(sum(fraction)-1.0)/2
##    print fraction

    wheel=[]

    pc=0

    for f in fraction:
        Np=int(f*Lwheel)
        for i in range(Np):
            wheel.append(pc)
        pc+=1

    return wheel

        
F1=F0[:]    #Copia de F0        

def nextgeneration(b):
    display.clear_output(wait=True)
    display.display(button)
    F0.sort(key=cmp_to_key(compare_chromosomes) )
    print( "Best solution so far:")
    print( "f(",decode_chromosome(F0[0]),")= ", f([decode_chromosome(F0[0])]) )
                                                                    
    #elitism, the two best chromosomes go directly to the next generation
    F1[0]=F0[0]
    F1[1]=F0[1]

    for i in range(0,int((N_chromosomes-2)/2)):
        roulette=create_wheel()
        #Two parents are selected
        p1=random.choice(roulette)
        p2=random.choice(roulette)
        #Two descendants are generated
        o1=F0[p1][0:crossover_point]
        o1.extend(F0[p2][crossover_point:L_chromosome])
        o2=F0[p2][0:crossover_point]
        o2.extend(F0[p1][crossover_point:L_chromosome])
        
        #Each descendant is mutated with probability prob_m
        if random.random() < prob_m:
            o1[int(round(random.random()*(L_chromosome-1)))]^=1   #XOR con 1 == NOT
        if random.random() < prob_m:
            o2[int(round(random.random()*(L_chromosome-1)))]^=1   #XOR con 1 == NOT
        #The descendants are added to F1
        F1[2+2*i]=o1
        F1[3+2*i]=o2


    graph_population(F1)
    #The generation replaces the old one
    F0[:]=F1[:]


def graph_f():
    xini=-5
    xfin=5

    rastrigin = ot.PythonFunction(2, 1, f)

    rastrigin = ot.MemoizeFunction(rastrigin)

    lowerbound = [xini] * 2
    upperbound = [xfin] * 2
    bounds = ot.Interval(lowerbound, upperbound)

    graph = rastrigin.draw(lowerbound, upperbound, [100] * 2)
    graph.setTitle("Rastrigin function")
    view = viewer.View(graph, legend_kw={"bbox_to_anchor": (1, 1), "loc": "upper left"})
    view.getFigure().tight_layout()


def graph_population(F):
    x=list(map(decode_chromosome,F))
    graph_f()
    plt.plot(x,y_population,'go')


button=create_button()
button.on_click(nextgeneration)
display.display(button)


x=list(map(decode_chromosome,F0))   #Aplico la funcion decode_chromosome al arreglo de cromosomas F0

y_population=np.zeros([N_chromosomes,N_chromosomes])  #Matriz de 10x10 con 0's

graph_f()
plt.plot(x,y_population,'go')
F0.sort(  key=cmp_to_key(compare_chromosomes))
evaluate_chromosomes()
