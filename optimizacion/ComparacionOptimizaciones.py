from datetime import datetime

from IPython.display import display, HTML
import dataframe_image as dfi
from modelos.ModeloV5 import Modelo as ModeloMioV5
from statistics import mean
from modelos.ModeloV4 import Modelo as ModeloMioV4
from modelos.ModeloOptimizacionNucleosActivosPorSubp import ModeloOptimizacionNucleosActivosPorSubp
from modelos.ModeloOptimizacionReparto import ModeloOptimizacionReparto as ModeloOptimizacionReparto
from modelos.ModeloOptimizacionRepartoPorSubp import ModeloOptimizacionRepartoPorSubp
from modelos.ModeloV6 import Modelo as ModeloMioV6

from Graficas import mostrarComparativa, mostrarCalculoCostes, mostrarComparativaConjuntaOptimizacion
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
matrizCluster = []
matrizCluster.append(2)  # Nodo 1
matrizCluster.append(2)  # Nodo 2
matrizCluster.append(2)  # Nodo 3

mejorIndividuo = []
mejorFitness = -1.0

OPTIMIZAR_REPARTO_POR_NUM_SUBP = False
importarModelo = True
importarOptimizacion = True
mostrarComparativaConjunta=True
mostrarConfiguracionOptima1=True
numModelos = 10


ahora = datetime.now()
cadena = ahora.strftime("%d_%m_%Y_%H_%M")

# nombreWorkspace = 'workspc_27N_2020'
# nombreWorkspace = 'workspc_13_marzo2019'
nombreWorkspace = '../workspc_30N'

'''
if importarModelo:
    direccionFicheroModelo = "ficherosModelos/ModeloV4_30_06_2021_18_58.mod"
    modelo = ModeloMioV4.importarModelo(direccionFicheroModelo)
    mejorIndividuoDado = modelo.construirIndividuo()
'''
if importarModelo:
    # direccionFicheroModelo = "ficherosModelos/ModeloV6_02_08_2021_19_27.mod"
    # direccionFicheroModelo = "ficherosModelos/ModeloV8_04_08_2021_20_31.mod"
    direccionFicheroModelo = "../ficherosModelos/ModeloV6_19_08_2021_19_47.mod"
    modelo = ModeloMioV6.importarModelo(direccionFicheroModelo)
    mejorIndividuoDado = modelo.construirIndividuo()
    modelo = ModeloMioV4(nombreWorkspace, matrizCluster, nombreModelo="ModeloV4")
    modelo.introducirParametros(mejorIndividuoDado, esGen=True, datosPotenciaDisp=True)
    mejorIndividuoDado = modelo.construirIndividuo()


else:
    modelo = ModeloMioV6(nombreWorkspace, matrizCluster, nombreModelo="ModeloV6")
    mejorIndividuo, mejorFitness = modelo.ajustarParametros()
    modelo = ModeloMioV4(nombreWorkspace, matrizCluster, nombreModelo="ModeloV4")
    modelo.introducirParametros(mejorIndividuo, esGen=True, datosPotenciaDisp=True)
    mejorIndividuoDado = modelo.construirIndividuo()


# Optimización núcleos activos por subpoblaciones:
dataframe = pd.read_csv('../OptimNucleosPorSubpModelOptimizacion_22_08_2021_23_18.csv')
numMaxSubp = 32
costesComputadosTiempo = []
costesComputadosEnergia = []
costesOptimizacionNucleosPorSubp = dict()

if mostrarConfiguracionOptima1:
    configuracionesOptimas = []

for numSubp in range(1, numMaxSubp + 1):
    dataframeSubp = dataframe[dataframe['0'] == numSubp]
    escala = MinMaxScaler()
    # escala = StandardScaler()
    escala = escala.fit(dataframeSubp.iloc[:, -2:])
    fitnessNormalizados = escala.transform(dataframeSubp.iloc[:, -2:])

    indices = np.arange(dataframeSubp.shape[0])
    fitnessNormalizados = list(zip(indices, fitnessNormalizados))
    mejorFitness = min(fitnessNormalizados, key=lambda fitness: fitness[1][0] + fitness[1][1])
    mejorIndividuo = list(dataframeSubp.iloc[mejorFitness[0], 2:-2])

    modelo = ModeloOptimizacionNucleosActivosPorSubp(nombreWorkspace, matrizCluster, numSubp,
                                                     nombreModelo="ModelOptimizacion")

    modelo.introducirParametros(mejorIndividuoDado, esGen=True)
    if mostrarConfiguracionOptima1:
        numNucleos = []
        for i in range(0, len(modelo.matrizCluster)):
            for j in range(0, modelo.matrizCluster[i]):
                numNucleos.append(str(int(modelo.datos['P_' + str(i + 1) + '_' + str(j + 1)])))

    costesReales, costesComputados = modelo.calcularFitness(mejorIndividuo, devolverResultados=True)

    if mostrarConfiguracionOptima1:
        nucleosActivos = [str(numSubp)]
        k = 0
        for i in range(0, len(modelo.matrizCluster)):
            for j in range(0, modelo.matrizCluster[i]):
                nucleosActivos.append(
                    str(modelo.datos['P_' + str(i + 1) + '_' + str(j + 1)]) + '/' + numNucleos[k]
                )
                k = k + 1
        configuracionesOptimas.append(nucleosActivos)

    costesComputadosTiempo.append(float(costesComputados[0]['T']))
    costesComputadosEnergia.append(float(costesComputados[0]['energy']))

if mostrarConfiguracionOptima1:
    columnas = ['N_Spop', 'Pa_1_1', 'Pa_1_2', 'Pa_2_1', 'Pa_2_2', 'Pa_3_1', 'Pa_3_2']
    d = pd.DataFrame(data=configuracionesOptimas, columns=columnas)
    # print(d.style)
    # display(d.style)
    # d.to_csv('configuracionesOpt1.csv')
    dfi.export(d, 'configuracionOptima.png')

costesOptimizacionNucleosPorSubp['T'], costesOptimizacionNucleosPorSubp['E'] = \
        costesComputadosTiempo, costesComputadosEnergia

# Optimización reparto por subpoblaciones:
costesOptimizacionRepartoPorSubp = dict()
dataframe = pd.read_csv('../OptimRepartoNucleosPorSubpModelOptimizacion_23_08_2021_16_29.csv')
numMaxSubp = 32
costesComputadosTiempo = []
costesComputadosEnergia = []

for numSubp in range(1, numMaxSubp + 1):
    dataframeSubp = dataframe[dataframe['0'] == numSubp]
    escala = MinMaxScaler()
    # escala = StandardScaler()
    escala = escala.fit(dataframeSubp.iloc[:, -2:])
    fitnessNormalizados = escala.transform(dataframeSubp.iloc[:, -2:])

    indices = np.arange(dataframeSubp.shape[0])
    fitnessNormalizados = list(zip(indices, fitnessNormalizados))
    mejorFitness = min(fitnessNormalizados, key=lambda fitness: fitness[1][0] + fitness[1][1])
    mejorIndividuo = list(dataframeSubp.iloc[mejorFitness[0], 2:-2])

    modelo = ModeloOptimizacionRepartoPorSubp(nombreWorkspace, matrizCluster, numSubp,
                                                     nombreModelo="ModelOptimizacion")

    modelo.introducirParametros(mejorIndividuoDado, esGen=True)
    costesReales, costesComputados = modelo.calcularFitness(mejorIndividuo, devolverResultados=True)
    costesComputadosTiempo.append(float(costesComputados[0]['T']))
    costesComputadosEnergia.append(float(costesComputados[0]['energy']))

modelo = ModeloOptimizacionReparto(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")

costesOptimizacionRepartoPorSubp['T'], costesOptimizacionRepartoPorSubp['E'] = \
    costesComputadosTiempo, costesComputadosEnergia

if mostrarComparativaConjunta:
    optimizacion1 = {
        'nombre': "ActCorePerSubpOpt",
        'costes': costesOptimizacionNucleosPorSubp
    }
    optimizacion2 = {
        'nombre': "ActCoreSchedPerSubpOpt",
        'costes': costesOptimizacionRepartoPorSubp
    }


mostrarComparativaConjuntaOptimizacion(optimizacion1, optimizacion2, nombreWorkspace=nombreWorkspace)
print('ok')

