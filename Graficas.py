import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scipy

from sklearn.preprocessing import MinMaxScaler

from modelos.ModeloV1 import Modelo as ModeloMioV1
from modelos.ModeloV2 import Modelo as ModeloMioV2
from modelos.ModeloV3 import Modelo as ModeloMioV3
from modelos.ModeloV4 import Modelo as ModeloMioV4
from modelos.ModeloV6 import Modelo as ModeloMioV6
from modelos.ModeloV8 import Modelo as ModeloMioV8

from statistics import stdev, mean

# Función para comparar los resultados de predicción con los datos experimentales:
def mostrarComparativa(modelo, individuo, costesComputadosTiempo=[], costesComputadosEnergia=[]):

    costesRealesTiempo = []
    costesRealesEnergia = []
    if (costesComputadosTiempo == []) and (costesComputadosEnergia == []):
        costesReales, costesComputados = modelo.calcularFitness(individuo, devolverResultados=True)
        costesComputadosTiempo = []
        costesComputadosEnergia = []

        for i in range(0, len(costesReales)):
            costesRealesTiempo.append(float(costesReales[i]['T']))
            costesComputadosTiempo.append(float(costesComputados[i]['T']))
            costesRealesEnergia.append(float(costesReales[i]['energy']))
            costesComputadosEnergia.append(float(costesComputados[i]['energy']))
    else:
        costesRealesTiempo = list(map(lambda x: float(x/1000), modelo.datos["time_v1v2v3_all"][:, 2]))
        costesRealesEnergia = list(map(lambda x: float(x),modelo.datos["energ_v1v2v3_all"][:, 2]))

    # numSubpoblaciones = np.arange(1, modelo.numMaxSubpoblaciones+1)
    numSubpoblaciones = np.arange(1, len(costesRealesEnergia)+1)

    anchuraBarra = 0.35
    plt.figure()
    plt.bar(numSubpoblaciones - anchuraBarra, costesRealesTiempo, width=anchuraBarra, align="center", label="Real")
    plt.bar(numSubpoblaciones, costesComputadosTiempo, width=anchuraBarra, align="center", label="Computado")
    plt.ylabel('Tiempo')
    plt.title('Tiempo real vs Tiempo computado')

    plt.legend(loc='best')
    plt.xlabel('Número de subpoblaciones')
    plt.xticks(numSubpoblaciones + anchuraBarra / 2, numSubpoblaciones)

    # plt.show()

    anchuraBarra = 0.35
    plt.figure()
    plt.bar(numSubpoblaciones - anchuraBarra, costesRealesEnergia, width=anchuraBarra, align="center", label="Real")
    plt.bar(numSubpoblaciones, costesComputadosEnergia, width=anchuraBarra, align="center", label="Computado")
    plt.ylabel('Energía')
    plt.title('Energía real vs Energía computada')

    plt.legend(loc='best')
    plt.xlabel('Número de subpoblaciones')
    plt.xticks(numSubpoblaciones + anchuraBarra / 2, numSubpoblaciones)

    plt.show()

# Función para mostrar una comparativa de las predicciones de los modelos dados con los datos experimentales:
def mostrarComparativaTodosModelos(direccionesModelos, nombreWorkspace = 'workspc_30N', graficarModeloV0=False,
        nombreFicheroExportado=None):
    matrizCluster = []
    matrizCluster.append(2)  # Nodo 1
    matrizCluster.append(2)  # Nodo 2
    matrizCluster.append(2)  # Nodo 3

    costesComputadosTiempoModelos = []
    costesComputadosEnergiaModelos = []

    nombresModelos = list(map(lambda m: m['nombre'], direccionesModelos))

    columnasDataframe = ["Modelo"]
    for k in range(1, 33):
        columnasDataframe.append("T_cluster_"+str(k))
    for k in range(1, 33):
        columnasDataframe.append("E_cluster_"+str(k))

    datosDataframe = []

    for m in direccionesModelos:
        nombreModelo = m['nombre']
        direccionDatos = m['direccionDatos']
        dataframe = pd.read_csv(direccionDatos)
        escala = MinMaxScaler()
        escala = escala.fit(dataframe.iloc[:,-2:])
        fitnessNormalizados = escala.transform(dataframe.iloc[:,-2:])

        indices = np.arange(dataframe.shape[0])
        fitnessNormalizados = list(zip(indices, fitnessNormalizados))
        mejorFitness = min(fitnessNormalizados, key=lambda fitness: fitness[1][0] + fitness[1][1])
        mejorIndividuo = list(dataframe.iloc[mejorFitness[0], 1:-2])


        if nombreModelo == "V1":
            modelo = ModeloMioV1(nombreWorkspace, matrizCluster, nombreModelo=nombreModelo)

        if nombreModelo == "V2":
            modelo = ModeloMioV2(nombreWorkspace, matrizCluster, nombreModelo=nombreModelo)

        if nombreModelo == "V6" or nombreModelo == "Proposed model":
            modelo = ModeloMioV6(nombreWorkspace, matrizCluster, nombreModelo=nombreModelo)

        if nombreModelo == "V8":
            modelo = ModeloMioV8(nombreWorkspace, matrizCluster, nombreModelo=nombreModelo)

        costesReales, costesComputados = modelo.calcularFitness(mejorIndividuo, devolverResultados=True)
        costesComputadosTiempo = list(map(lambda coste: float(coste['T']), costesComputados))
        costesComputadosEnergia = list(map(lambda coste: float(coste['energy']), costesComputados))

        costesComputadosTiempoModelos.append(costesComputadosTiempo)
        costesComputadosEnergiaModelos.append(costesComputadosEnergia)

        fila = [nombreModelo]
        fila.extend(costesComputadosTiempo)
        fila.extend(costesComputadosEnergia)
        datosDataframe.append(fila)


    costesRealesTiempo = list(map(lambda x: float(x/1000), modelo.datos["time_v1v2v3_all"][:, 2]))
    costesRealesEnergia = list(map(lambda x: float(x),modelo.datos["energ_v1v2v3_all"][:, 2]))

    fila = ["Experimental"]
    fila.extend(costesRealesTiempo)
    fila.extend(costesRealesEnergia)
    datosDataframe.append(fila)


    if graficarModeloV0:
        # Leemos los costes computados:
        datos = scipy.loadmat(nombreWorkspace)
        costesComputadosTiempo = list(datos['TimeV3'].flatten())
        costesComputadosEnergia = list(datos['EnergyV3'].flatten())

        costesComputadosTiempoModelos.append(costesComputadosTiempo)
        costesComputadosEnergiaModelos.append(costesComputadosEnergia)
        # nombresModelos.append('Escobar et al')
        nombreModelo = 'Escobar et al (2019)'
        nombresModelos.append(nombreModelo)

        fila = [nombreModelo]
        fila.extend(costesComputadosTiempo)
        fila.extend(costesComputadosEnergia)
        datosDataframe.append(fila)

    dataframe = pd.DataFrame(data=datosDataframe, columns=columnasDataframe)

    if not nombreFicheroExportado is None:
        dataframe.to_csv(nombreFicheroExportado, index=False)


    numSubpoblaciones = np.arange(1, 33)
    numBarras = len(direccionesModelos) + 1
    numBarras = numBarras + 1 if graficarModeloV0 else numBarras

    anchuraBarra =0.75
    anchuraBarra = anchuraBarra / numBarras
    plt.figure()
    posicion = numSubpoblaciones - anchuraBarra
    plt.bar(posicion, costesRealesTiempo, width=anchuraBarra, align="center", label="Experimental")
    for nombreModelo, costesComputadosTiempo in zip(nombresModelos, costesComputadosTiempoModelos):
        posicion = posicion + anchuraBarra
        plt.bar(posicion, costesComputadosTiempo, width=anchuraBarra, align="center", label=nombreModelo)

    plt.ylabel('Runtime (seconds)')
    # plt.title('Experimental runtime VS Predicted runtime')
    plt.title('Experimental and predicted runtime')

    plt.legend(loc='best')
    plt.xlabel('Number of subpopulations')
    plt.xticks(numSubpoblaciones + anchuraBarra / 2, numSubpoblaciones)

    # plt.show()

    plt.figure()
    posicion = numSubpoblaciones - anchuraBarra
    plt.bar(posicion, costesRealesEnergia, width=anchuraBarra, align="center", label="Experimental")
    for nombreModelo, costesComputadosEnergia in zip(nombresModelos, costesComputadosEnergiaModelos):
        posicion = posicion + anchuraBarra
        plt.bar(posicion, costesComputadosEnergia, width=anchuraBarra, align="center", label=nombreModelo)

    plt.ylabel('Energy consumption (watts · hour)')
    # plt.title('Experimental energy consumption VS Predicted energy consumption')
    plt.title('Experimental and predicted energy consumption')

    plt.legend(loc='best')
    # plt.xlabel('Número de subpoblaciones')
    plt.xlabel('Number of subpopulations')
    plt.xticks(numSubpoblaciones + anchuraBarra / 2, numSubpoblaciones)

    plt.show()

# Función para mostrar una comparativa de las optimizaciones de los dos modelos dados con los datos experimentales:
def mostrarComparativaConjuntaOptimizacion(optimizacion1, optimizacion2, nombreWorkspace = 'workspc_30N'):
    matrizCluster = []
    matrizCluster.append(2)  # Nodo 1
    matrizCluster.append(2)  # Nodo 2
    matrizCluster.append(2)  # Nodo 3
    modelo = ModeloMioV6(nombreWorkspace, matrizCluster, nombreModelo="ModeloV6")
    costesRealesTiempo = list(map(lambda x: float(x / 1000), modelo.datos["time_v1v2v3_all"][:, 2]))
    costesRealesEnergia = list(map(lambda x: float(x), modelo.datos["energ_v1v2v3_all"][:, 2]))

    mediaAhorroTiempo1 = 0.0
    mediaAhorroEnergia1 = 0.0

    mediaAhorroTiempo2 = 0.0
    mediaAhorroEnergia2 = 0.0

    diferenciasTiempo1 = np.array(optimizacion1['costes']['T']) - np.array(costesRealesTiempo)
    mediaDiferenciasTiempo1 = np.mean(diferenciasTiempo1)
    diferenciasTiempo2 = np.array(optimizacion2['costes']['T']) - np.array(costesRealesTiempo)
    mediaDiferenciasTiempo2 = np.mean(diferenciasTiempo2)

    diferenciasEnergia1 = np.array(optimizacion1['costes']['E']) - np.array(costesRealesEnergia)
    mediaDiferenciasEnergia1 = np.mean(diferenciasEnergia1)
    diferenciasEnergia2 = np.array(optimizacion2['costes']['E']) - np.array(costesRealesEnergia)
    mediaDiferenciasEnergia2 = np.mean(diferenciasEnergia2)

    print('Medias de tiempo y energia con opt 1: ', mediaDiferenciasTiempo1, "/ ", np.mean(costesRealesTiempo),
          ", ", mediaDiferenciasEnergia1, "/ ", np.mean(costesRealesEnergia))
    print('Medias de tiempo y energia con opt 2: ', mediaDiferenciasTiempo2, "/ ", np.mean(costesRealesTiempo),
          ", ", mediaDiferenciasEnergia2, "/ ", np.mean(costesRealesEnergia))

    numSubpoblaciones = np.arange(1, 33)
    numBarras = 2 + 1

    anchuraBarra =0.75
    anchuraBarra = anchuraBarra / numBarras
    plt.figure()
    posicion = numSubpoblaciones - anchuraBarra
    plt.bar(posicion, costesRealesTiempo, width=anchuraBarra, align="center", label="Experimental")
    posicion = posicion + anchuraBarra
    plt.bar(posicion, optimizacion1['costes']['T'], width=anchuraBarra, align="center", label=optimizacion1['nombre'])
    posicion = posicion + anchuraBarra
    plt.bar(posicion, optimizacion2['costes']['T'], width=anchuraBarra, align="center", label=optimizacion2['nombre'])
    plt.ylabel('Runtime')
    plt.title('Experimental runtime VS Optimized runtime')

    plt.legend(loc='best')
    plt.xlabel('Number of subpopulations')
    plt.xticks(numSubpoblaciones + anchuraBarra / 2, numSubpoblaciones)


    plt.figure()
    posicion = numSubpoblaciones - anchuraBarra
    plt.bar(posicion, costesRealesEnergia, width=anchuraBarra, align="center", label="Experimental")
    posicion = posicion + anchuraBarra
    plt.bar(posicion, optimizacion1['costes']['E'], width=anchuraBarra, align="center", label=optimizacion1['nombre'])
    posicion = posicion + anchuraBarra
    plt.bar(posicion, optimizacion2['costes']['E'], width=anchuraBarra, align="center", label=optimizacion2['nombre'])

    plt.ylabel('Energy consumption')
    plt.title('Experimental energy consumption VS Optimized energy consumption')

    plt.legend(loc='best')
    plt.xlabel('Number of subpopulations')
    plt.xticks(numSubpoblaciones + anchuraBarra / 2, numSubpoblaciones)

    plt.show()