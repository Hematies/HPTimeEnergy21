from datetime import datetime

from modelos.ModeloV5 import Modelo as ModeloMioV5
from statistics import mean
from modelos.ModeloV4 import Modelo as ModeloMioV4
from modelos.ModeloOptimizacionNucleosActivos import ModeloOptimizacionNucleosActivos
from modelos.ModeloOptimizacionNucleosActivosPorSubp import ModeloOptimizacionNucleosActivosPorSubp
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

# En el presente script se aplica optimización vía DCT.
OPTIMIZAR_NUM_NUCLEOS_POR_NUM_SUBP = True # ¿Una optimización para todos los casos o para cada num. de subpoblaciones?
importarModelo = True # ¿Importar modelo ya ajustado?
importarOptimizacion = True # ¿Importar modelo ya optimizado?
mostrarComparativaConjunta = True # ¿Mostrar una comparativa de las dos optimizaciones?


# nombreWorkspace = 'workspc_27N_2020'
# nombreWorkspace = 'workspc_13_marzo2019'
nombreWorkspace = 'workspc_30N'

numModelos = 10

ahora = datetime.now()
cadena = ahora.strftime("%d_%m_%Y_%H_%M")

if importarModelo:
    # direccionFicheroModelo = "ficherosModelos/ModeloV6_02_08_2021_19_27.mod"
    # direccionFicheroModelo = "ficherosModelos/ModeloV8_04_08_2021_20_31.mod"
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
    if not OPTIMIZAR_NUM_NUCLEOS_POR_NUM_SUBP:
        for i in range(0, numModelos):
            print('========')
            print('VAMOS CON EL MODELO ',i)

            modelo = ModeloOptimizacionNucleosActivos(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")
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
        dataframe.to_csv('OptimNucleos'+modelo.nombreModelo + '_' + cadena + '.csv')

    else:
        for i in range(0, numModelos):
            print('========')
            print('VAMOS CON EL MODELO ', i)
            numMaxSubp = 32
            costesComputadosTiempo = []
            costesComputadosEnergia = []
            for numSubp in range(1, numMaxSubp+1):
                modelo = ModeloOptimizacionNucleosActivosPorSubp(nombreWorkspace, matrizCluster, numSubp, nombreModelo="ModelOptimizacion")
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
                modelo = ModeloOptimizacionNucleosActivos(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")
                mostrarComparativa(modelo, mejorIndividuo,
                                   costesComputadosTiempo=costesComputadosTiempo, costesComputadosEnergia=costesComputadosEnergia)

        # Exportamos los datos de las 10 ejecuciones:
        dataframe = pd.DataFrame(data=datos)
        dataframe.to_csv('OptimNucleosPorSubp' + modelo.nombreModelo + '_' + cadena + '.csv')

else:
    costesOptimizacionNucleos = dict()
    costesOptimizacionNucleosPorSubp = dict()
    if not OPTIMIZAR_NUM_NUCLEOS_POR_NUM_SUBP or mostrarComparativaConjunta:
        dataframe = pd.read_csv('OptimNucleosModelOptimizacion_22_08_2021_23_04.csv')
        escala = MinMaxScaler()
        # escala = StandardScaler()
        escala = escala.fit(dataframe.iloc[:, -2:])
        fitnessNormalizados = escala.transform(dataframe.iloc[:, -2:])

        indices = np.arange(dataframe.shape[0])
        fitnessNormalizados = list(zip(indices, fitnessNormalizados))
        mejorFitness = min(fitnessNormalizados, key=lambda fitness: fitness[1][0] + fitness[1][1])
        mejorIndividuo = list(dataframe.iloc[mejorFitness[0], 1:-2])

        modelo = ModeloOptimizacionNucleosActivos(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")
        modelo.introducirParametros(mejorIndividuoDado, esGen=True)

        if not mostrarComparativaConjunta:
            mostrarComparativa(modelo, mejorIndividuo)
        else:
            costes = modelo.calcularFitness(mejorIndividuo, devolverResultados=True)[1]
            costesOptimizacionNucleos['T'] = list(map(lambda coste: float(coste['T']), costes))
            costesOptimizacionNucleos['E'] = list(map(lambda coste: float(coste['energy']), costes))

    if OPTIMIZAR_NUM_NUCLEOS_POR_NUM_SUBP or mostrarComparativaConjunta:
        dataframe = pd.read_csv('OptimNucleosPorSubpModelOptimizacion_22_08_2021_23_18.csv')
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

            modelo = ModeloOptimizacionNucleosActivosPorSubp(nombreWorkspace, matrizCluster, numSubp,
                                                             nombreModelo="ModelOptimizacion")
            modelo.introducirParametros(mejorIndividuoDado, esGen=True)
            costesReales, costesComputados = modelo.calcularFitness(mejorIndividuo, devolverResultados=True)
            costesComputadosTiempo.append(float(costesComputados[0]['T']))
            costesComputadosEnergia.append(float(costesComputados[0]['energy']))

        modelo = ModeloOptimizacionNucleosActivos(nombreWorkspace, matrizCluster, nombreModelo="ModelOptimizacion")
        if not mostrarComparativaConjunta:
            mostrarComparativa(modelo, mejorIndividuo,
                           costesComputadosTiempo=costesComputadosTiempo,
                           costesComputadosEnergia=costesComputadosEnergia)
        else:
            costesOptimizacionNucleosPorSubp['T'], costesOptimizacionNucleosPorSubp['E'] = \
                costesComputadosTiempo, costesComputadosEnergia

    if mostrarComparativaConjunta:
        optimizacion1 = {
            'nombre': "ActCoreOpt",
            'costes': costesOptimizacionNucleos
        }
        optimizacion2 = {
            'nombre': "ActCorePerSubpOpt",
            'costes': costesOptimizacionNucleosPorSubp
        }
        mostrarComparativaConjuntaOptimizacion(optimizacion1, optimizacion2)

print('ok')

