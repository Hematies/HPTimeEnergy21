import math

# import matplotlib
# matplotlib.use('Agg')

from modelos.ModeloV1 import Modelo as ModeloMioV1
from modelos.ModeloV2 import Modelo as ModeloMioV2
from modelos.ModeloV3 import Modelo as ModeloMioV3
from modelos.ModeloV4 import Modelo as ModeloMioV4
from modelos.ModeloV6 import Modelo as ModeloMioV6
from modelos.ModeloV8 import Modelo as ModeloMioV8

import pandas as pd

from minisom import MiniSom
from sklearn.preprocessing import minmax_scale, scale,MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datetime import datetime
from statistics import stdev
from copy import copy

# 1) Definir la función de AG: una rutina que ejecuta el modelo dado y compara los resultados de energía y
# tiempo con los experimentales.
matrizCluster = []
matrizCluster.append(2)  # Nodo 1
matrizCluster.append(2)  # Nodo 2
matrizCluster.append(2)  # Nodo 3

# Elegimos qué modelos usar:
usarModeloV1 = False # NGEN=100, MU=200,
usarModeloV2 = False # NGEN=100, MU=200,
usarModeloV3 = False
usarModeloV4 = False
usarModeloV6 = True # NGEN=100, MU=100,
usarModeloV8 = False # ...

mejorIndividuo = []
mejorFitness = -1.0

# Núm. de modelos independientes:
numModelos = 20
# numModelos = 10
# numModelos = 5

modelos = []
mejoresIndividuos = []

# nombreWorkspace = '../workspc_27N_2020'
# nombreWorkspace = '../workspc_13_marzo2019'
nombreWorkspace = '../workspc_30N'

# En el presente script se evalúa la estabilidad de los parámetros.
importarDatos = False # ¿Importamos los datos de estabilidad ya calculados con anterioridad?

ahora = datetime.now()
cadena = ahora.strftime("%d_%m_%Y_%H_%M")

if not importarDatos:
    for i in range(0, numModelos):
        print('--- MODELO NUMERO ',i, ' ---')

        if usarModeloV1:
            modelo = ModeloMioV1(nombreWorkspace, matrizCluster, nombreModelo="ModeloV1")
            # modelo.introducirParametros(mejorIndividuo, esGen=True)
            mejorIndividuo, mejorFitness = modelo.ajustarParametros()
            print("===============")
            print("Fitness del mejor individuo a partir del modelo V1:", modelo.calcularFitness(mejorIndividuo))
            print("===============")

        if usarModeloV2:
            modelo = ModeloMioV2(nombreWorkspace, matrizCluster,nombreModelo="ModeloV2")
            if usarModeloV1:
                modelo.introducirParametros(mejorIndividuo, esGen=True)
            mejorIndividuo, mejorFitness = modelo.ajustarParametros()
            print("===============")
            print("Fitness del mejor individuo a partir del modelo V2:", modelo.calcularFitness(mejorIndividuo))
            print("===============")

        if usarModeloV3:
            modelo = ModeloMioV3(nombreWorkspace, matrizCluster, nombreModelo="ModeloV3")
            if usarModeloV1 or usarModeloV2:
                modelo.introducirParametros(mejorIndividuo, esGen=True)
            mejorIndividuo, mejorFitness = modelo.ajustarParametros()
            print("===============")
            print("Fitness del mejor individuo a partir del modelo V3:", modelo.calcularFitness(mejorIndividuo))
            print("===============")

        if usarModeloV4:
            modelo = ModeloMioV4(nombreWorkspace, matrizCluster, nombreModelo="ModeloV4")
            if usarModeloV1 or usarModeloV2 or usarModeloV3:
                modelo.introducirParametros(mejorIndividuo, esGen=True, datosPotenciaDisp=True)
            mejorIndividuo, mejorFitness = modelo.ajustarParametros()
            print("===============")
            print("Fitness del mejor individuo a partir del modelo V4:", modelo.calcularFitness(mejorIndividuo))
            print("===============")

        if usarModeloV6:
            modelo = ModeloMioV6(nombreWorkspace, matrizCluster, nombreModelo="ModeloV6")
            if usarModeloV1 or usarModeloV2 or usarModeloV3 or usarModeloV4:
                modelo.introducirParametros(mejorIndividuo, esGen=True, datosPotenciaDisp=True)
            mejorIndividuo, mejorFitness = modelo.ajustarParametros()
            print("===============")
            print("Fitness del mejor individuo a partir del modelo V6", modelo.calcularFitness(mejorIndividuo))
            print("===============")

        if usarModeloV8:
            modelo = ModeloMioV8(nombreWorkspace, matrizCluster, nombreModelo="ModeloV8")
            if usarModeloV1 or usarModeloV2 or usarModeloV3 or usarModeloV4 or usarModeloV6:
                modelo.introducirParametros(mejorIndividuo, esGen=True, datosPotenciaDisp=True)
            mejorIndividuo, mejorFitness = modelo.ajustarParametros()
            mejorFitness = modelo.calcularFitness(mejorIndividuo)
            print("===============")
            print("Fitness del mejor individuo a partir del modelo V8:", mejorFitness)
            print("===============")


        modelos.append(modelo)

        mejorIndividuo.append(mejorIndividuo.fitness.values[0])
        mejorIndividuo.append(mejorIndividuo.fitness.values[1])
        mejoresIndividuos.append(mejorIndividuo)

    # Construimos el dataframe de parámetros:

    dataframe = pd.DataFrame(data=mejoresIndividuos) # , columns=columnas)
    desviacion = dataframe.std(axis = 0)

    dataframe.to_csv('testEstabilidad_'+modelo.nombreModelo + '_' + cadena + '.csv')
else:

    dataframe = pd.read_csv('testEstabilidad_ModeloV6_17_08_2021_18_52.csv')

desviaciones = dataframe.std(axis = 0)
medias = dataframe.mean(axis=0)
desviacionesNormalizada = desviaciones / medias

print(desviacionesNormalizada.values)

print('ok')

