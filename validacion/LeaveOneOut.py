

from modelos.ModeloV5 import Modelo as ModeloMioV5
from modelos.ModeloV7 import Modelo as ModeloMioV7
from statistics import mean, stdev

import pandas as pd
import numpy as np

# nombreWorkspace = '../workspc_27N_2020'
# nombreWorkspace = '../workspc_13_marzo2019'
nombreWorkspace = '../workspc_30N'

matrizCluster = []
matrizCluster.append(2)  # Nodo 1
matrizCluster.append(2)  # Nodo 2
matrizCluster.append(2)  # Nodo 3

mejorIndividuo = []
mejorFitness = -1.0

# En el presente script se aplica validación por LOO-CV.
importarModelo = False # ¿Importamos un modelo ya ajustado vía LOO-CV?

modelo = ModeloMioV7(nombreWorkspace, matrizCluster, nombreModelo="ModeloV7")

if not importarModelo:
    resultados = modelo.aplicarCV(32, exportarModelos=True) # 32 folds para el LOO
else:
    dataframe = pd.read_csv('LOO_CV_ModeloV7_13_12_2021_14_55.csv')
    numColumnas = len(dataframe.columns)

tiemposReales = modelo.datos["time_v1v2v3_all"][:, 2]
energiasReales = modelo.datos["energ_v1v2v3_all"][:, 2]
dispersionTiempo = stdev(list(np.array(tiemposReales) / 1000)) # Está en milisegudos
dispersionEnergia = stdev(energiasReales)

print('Resultados de cada pliegue/partición (fold):')
dataframe_dummy = dataframe.iloc[:,-2:]
dataframe_dummy['23'] = dataframe_dummy['23'].apply(lambda x: x * dispersionTiempo)
dataframe_dummy['24'] = dataframe_dummy['24'].apply(lambda x: x * dispersionEnergia)
resultados = dataframe_dummy.values.tolist()
print(resultados)
resultadosTiempos = list(map(lambda resultado: resultado[0], resultados))
resultadosEnergias = list(map(lambda resultado: resultado[1], resultados))

print('Media de error en tiempo y en energía: ', mean(resultadosTiempos), ", ", mean(resultadosEnergias))

print('=====')

print('Resultados normalizados de cada pliegue/partición (fold):')
resultados = dataframe.iloc[:,-2:].values.tolist()
print(resultados)
resultadosTiempos = list(map(lambda resultado: resultado[0], resultados))
resultadosEnergias = list(map(lambda resultado: resultado[1], resultados))
print('Media de error normalizado en tiempo y en energía: ', mean(resultadosTiempos), ", ", mean(resultadosEnergias))



print('ok')

