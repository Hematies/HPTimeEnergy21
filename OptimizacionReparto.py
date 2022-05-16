from datetime import datetime

from modelos.ModeloV5 import Modelo as ModeloMioV5
from statistics import mean
from modelos.ModeloV4 import Modelo as ModeloMioV4
from modelos.ModeloOptimizacionReparto import ModeloOptimizacionReparto as ModeloOptimizacionReparto
from modelos.ModeloOptimizacionRepartoPorSubp import ModeloOptimizacionRepartoPorSubp
from modelos.ModeloV6 import Modelo as ModeloMioV6

from Graficas import mostrarComparativa, mostrarComparativaConjuntaOptimizacion
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
matrizCluster = []
matrizCluster.append(2)  # Nodo 1
matrizCluster.append(2)  # Nodo 2
matrizCluster.append(2)  # Nodo 3

mejorIndividuo = []
mejorFitness = -1.0

# En el presente script se aplica optimización vía DCT y reparto estático de carga de trabajo.
OPTIMIZAR_REPARTO_POR_NUM_SUBP = False# ¿Una optimización para todos los casos o para cada num. de subpoblaciones?
importarModelo = True # ¿Importar modelo ya ajustado?
importarOptimizacion = True # ¿Importar modelo ya optimizado?
mostrarComparativaConjunta = True # ¿Mostrar una comparativa de las dos optimizaciones?
numModelos = 10


ahora = datetime.now()
cadena = ahora.strftime("%d_%m_%Y_%H_%M")

# nombreWorkspace = 'workspc_27N_2020'
# nombreWorkspace = 'workspc_13_marzo2019'
nombreWorkspace = 'workspc_30N'

if importarModelo:
    direccionFicheroModelo = "ficherosModelos/ModeloV6_19_08_2021_19_47.mod"
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


datos = []
if not importarOptimizacion:
    if not OPTIMIZAR_REPARTO_POR_NUM_SUBP:
        for i in range(0, numModelos):
            print('========')
            print('VAMOS CON EL MODELO ', i)

            modelo = ModeloOptimizacionReparto(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")
            modelo.introducirParametros(mejorIndividuoDado, esGen=True)

            mejorIndividuo, mejorFitness = modelo.ajustarParametros()

            if numModelos == 1:
                mostrarComparativa(modelo, mejorIndividuo)


            fila = []
            fila.extend(mejorIndividuo)
            fila.extend(mejorFitness.values)
            datos.append(fila)

            # Exportamos los datos de las 10 ejecuciones:
        dataframe = pd.DataFrame(data=datos)
        dataframe.to_csv('OptimRepartoNucleos' + modelo.nombreModelo + '_' + cadena + '.csv')

    else:
        for i in range(0, numModelos):
            print('========')
            print('VAMOS CON EL MODELO ', i)
            numMaxSubp = 32
            costesComputadosTiempo = []
            costesComputadosEnergia = []
            for numSubp in range(1, numMaxSubp+1):
                modelo = ModeloOptimizacionRepartoPorSubp(nombreWorkspace, matrizCluster, numSubp, nombreModelo="ModelOptimizacion")
                modelo.introducirParametros(mejorIndividuoDado, esGen=True)

                mejorIndividuo, mejorFitness = modelo.ajustarParametros()

                costesReales, costesComputados = modelo.calcularFitness(mejorIndividuo, devolverResultados=True)
                costesComputadosTiempo.append(float(costesComputados[0]['T']))
                costesComputadosEnergia.append(float(costesComputados[0]['energy']))

                fila = [numSubp]
                fila.extend(mejorIndividuo)
                fila.extend(mejorFitness.values)
                datos.append(fila)

            if numModelos == 1:
                modelo = ModeloOptimizacionReparto(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")
                mostrarComparativa(modelo, mejorIndividuo,
                                   costesComputadosTiempo=costesComputadosTiempo, costesComputadosEnergia=costesComputadosEnergia)

        # Exportamos los datos de las 10 ejecuciones:
        dataframe = pd.DataFrame(data=datos)
        dataframe.to_csv('OptimRepartoNucleosPorSubp' + modelo.nombreModelo + '_' + cadena + '.csv')
else:
    costesOptimizacionReparto = dict()
    costesOptimizacionRepartoPorSubp = dict()
    if not OPTIMIZAR_REPARTO_POR_NUM_SUBP or mostrarComparativaConjunta:
        dataframe = pd.read_csv('OptimRepartoNucleosModelOptimizacion_23_08_2021_17_23.csv')
        escala = MinMaxScaler()
        # escala = StandardScaler()
        escala = escala.fit(dataframe.iloc[:, -2:])
        fitnessNormalizados = escala.transform(dataframe.iloc[:, -2:])

        indices = np.arange(dataframe.shape[0])
        fitnessNormalizados = list(zip(indices, fitnessNormalizados))
        mejorFitness = min(fitnessNormalizados, key=lambda fitness: fitness[1][0] + fitness[1][1])
        mejorIndividuo = list(dataframe.iloc[mejorFitness[0], 1:-2])

        modelo = ModeloOptimizacionReparto(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")
        modelo.introducirParametros(mejorIndividuoDado, esGen=True)

        if not mostrarComparativaConjunta:
            mostrarComparativa(modelo, mejorIndividuo)
        else:
            costes = modelo.calcularFitness(mejorIndividuo, devolverResultados=True)[1]
            costesOptimizacionReparto['T'] = list(map(lambda coste: float(coste['T']), costes))
            costesOptimizacionReparto['E'] = list(map(lambda coste: float(coste['energy']), costes))

    if OPTIMIZAR_REPARTO_POR_NUM_SUBP or mostrarComparativaConjunta:
        dataframe = pd.read_csv('OptimRepartoNucleosPorSubpModelOptimizacion_23_08_2021_16_29.csv')
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
        if not mostrarComparativaConjunta:
            mostrarComparativa(modelo, mejorIndividuo,
                               costesComputadosTiempo=costesComputadosTiempo,
                               costesComputadosEnergia=costesComputadosEnergia)
        else:
            costesOptimizacionRepartoPorSubp['T'], costesOptimizacionRepartoPorSubp['E'] = \
                costesComputadosTiempo, costesComputadosEnergia

    if mostrarComparativaConjunta:
        optimizacion1 = {
            'nombre': "ActCoreSchedOpt",
            'costes': costesOptimizacionReparto
        }
        optimizacion2 = {
            'nombre': "ActCoreSchedPerSubpOpt",
            'costes': costesOptimizacionRepartoPorSubp
        }
        mostrarComparativaConjuntaOptimizacion(optimizacion1, optimizacion2)

print('ok')

