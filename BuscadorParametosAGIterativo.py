import math

from modelos.ModeloV1 import Modelo as ModeloMioV1
from modelos.ModeloV2 import Modelo as ModeloMioV2
from modelos.ModeloV3 import Modelo as ModeloMioV3
from modelos.ModeloV4 import Modelo as ModeloMioV4
from modelos.ModeloV5 import Modelo as ModeloMioV5
from modelos.ModeloV6 import Modelo as ModeloMioV6
from Graficas import mostrarComparativa


from modelos.ModeloV8 import Modelo as ModeloMioV8


from minisom import MiniSom
from sklearn.preprocessing import minmax_scale, scale,MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matrizCluster = []
matrizCluster.append(2)  # Nodo 1
matrizCluster.append(2)  # Nodo 2
matrizCluster.append(2)  # Nodo 3

# Elegimos qué modelos usar:
usarModeloV1 = False
usarModeloV2 = False
usarModeloV3 = False
usarModeloV4 = False
usarModeloV6 = True
usarModeloV8 = False

mejorIndividuo = []
mejorFitness = -1.0

# nombreWorkspace = 'workspc_27N_2020'
# nombreWorkspace = 'workspc_13_marzo2019'
nombreWorkspace = 'workspc_30N'

# Este script aplica ajuste (o simple importación) de un modelo y muestra los resultados de visualizar el espacio de
# configuraciones al optimizar en núcleos activos:
importarModelo = False # ¿Importamos un modelo ya ajustado?
importarDatosSimulacion = False # ¿Importamos el espacio de configuraciones?
direccionFicheroModelo = "ficherosModelos/ModeloV6_19_08_2021_19_47.mod"

if (not importarModelo) or (direccionFicheroModelo == None):
    if usarModeloV1:
        modelo = ModeloMioV1(nombreWorkspace, matrizCluster, nombreModelo="ModeloV1")
        # modelo.introducirParametros(mejorIndividuo, esGen=True)
        mejorIndividuo, mejorFitness = modelo.ajustarParametros()
        print("===============")
        print("Fitness del mejor individuo a partir del modelo V1:", modelo.calcularFitness(mejorIndividuo))
        print("===============")

    if usarModeloV2:
        modelo = ModeloMioV2(nombreWorkspace, matrizCluster, nombreModelo="ModeloV2")
        # modelo = ModeloMioV5(nombreWorkspace, matrizCluster, nombreModelo="ModeloV2")
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
        print("Fitness del mejor individuo a partir del modelo V6:", modelo.calcularFitness(mejorIndividuo))
        print("===============")

    if usarModeloV8:
        modelo = ModeloMioV8(nombreWorkspace, matrizCluster, nombreModelo="ModeloV8")
        if usarModeloV1 or usarModeloV2 or usarModeloV3 or usarModeloV4 or usarModeloV6:
            modelo.introducirParametros(mejorIndividuo, esGen=True, datosPotenciaDisp=True)
        mejorIndividuo, mejorFitness = modelo.ajustarParametros()
        print("===============")
        print("Fitness del mejor individuo a partir del modelo V8:", modelo.calcularFitness(mejorIndividuo))
        print("===============")

    # Exportamos el modelo:
    modelo.exportarModelo()
else:
    if usarModeloV1:
        modelo = ModeloMioV1.importarModelo(direccionFicheroModelo)
    elif usarModeloV2:
        modelo = ModeloMioV2.importarModelo(direccionFicheroModelo)
    elif usarModeloV4:
        modelo = ModeloMioV4.importarModelo(direccionFicheroModelo)
    elif usarModeloV6:
        modelo = ModeloMioV6.importarModelo(direccionFicheroModelo)
    else:
        modelo = ModeloMioV8.importarModelo(direccionFicheroModelo)

    mejorIndividuo = modelo.construirIndividuo()


modelo.mostrarValoresIndividuo(mejorIndividuo)

mostrarComparativa(modelo, mejorIndividuo)

direccionDatosSimulacion = 'datosSimulacion/ModeloV4_27_08_2021_00_49.csv'
modelo = ModeloMioV4(nombreWorkspace, matrizCluster, nombreModelo="ModeloV4")
modelo.introducirParametros(mejorIndividuo, esGen=True, datosPotenciaDisp=True)

if importarDatosSimulacion and (direccionDatosSimulacion != None):
    modelo.importarDatosSimulacion(direccionDatosSimulacion)
    resultadosSimulacion = modelo.datosSimulacion
else:
    resultadosSimulacion = modelo.simulacionCostesNumNucDisp(parametros=mejorIndividuo)
    modelo.exportarDatosSimulacion()

# Cambiamos el nombre de las columnas al inglés:
resultadosSimulacion = resultadosSimulacion.rename(
    columns={'numSubpoblaciones':'N_Spop', 'costeTemporal':'T_cluster', 'costeEnergetico':'E_cluster'})

resultadosSimulacion = resultadosSimulacion.drop(resultadosSimulacion.columns[0], axis=1) # Un apaño para quitar índices
datosBrutos = resultadosSimulacion.values
escala = StandardScaler()
# escala = MinMaxScaler()
escala = escala.fit(datosBrutos)
datosNormalizados = escala.transform(datosBrutos)

# Vamos a enrear con el SOM:
# anchuraNeuronas = 10
anchuraNeuronas = 12
# anchuraNeuronas = 16
som = MiniSom(anchuraNeuronas, anchuraNeuronas, len(datosNormalizados[0]),
              neighborhood_function='gaussian')
              # sigma=1.5,
              # random_seed=1)

som.pca_weights_init(datosNormalizados)
som.train_random(datosNormalizados, 1000, verbose=True)

W = som.get_weights()
plt.figure()
for i, f in enumerate(resultadosSimulacion.columns):
    plt.subplot(3, 3, i+1)
    plt.title(f)
    plt.pcolor(W[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(anchuraNeuronas+1))
    plt.yticks(np.arange(anchuraNeuronas+1))
plt.tight_layout(pad=0.5)
# plt.show()

# Vamos a enrear aún más:
Z = np.zeros((anchuraNeuronas, anchuraNeuronas))
plt.figure()
for i in np.arange(som._weights.shape[0]):
    for j in np.arange(som._weights.shape[1]):
        feature = np.argmax(W[i, j, :])
        plt.plot([j + .5], [i + .5], 'o', color='C' + str(feature),
                 marker='s', markersize=24)

legend_elements = [Patch(facecolor='C' + str(i),
                         edgecolor='w',
                         label=f) for i, f in enumerate(resultadosSimulacion.columns)]

plt.legend(handles=legend_elements,
           loc='center left',
           bbox_to_anchor=(1, .95))

plt.xlim([0, anchuraNeuronas])
plt.ylim([0, anchuraNeuronas])

# Más enreo aún:
plt.figure()
fuente = {
    'size': 8
    }
for i, f in enumerate(resultadosSimulacion.columns):
    for j in np.arange(som._weights.shape[0]):
        for k in np.arange(som._weights.shape[1]):
            plt.subplot(3, 3, i+1)
            plt.title(f)
            plt.text(j,k,str(int(round(escala.inverse_transform(W)[j,k,i]))), fontdict=fuente)
            # plt.text(j, k, str(int(round(escala.inverse_transform(W[j])[k, i]))), fontdict=fuente)
            plt.pcolor(W[:, :, i].T, cmap='coolwarm')
            plt.xticks(np.arange(anchuraNeuronas+1))
            plt.yticks(np.arange(anchuraNeuronas+1))
plt.tight_layout(pad=0.5)
# plt.show()

# Idea: Aplicar K-Medias sobre los vectores (costeTemp, costeEnerg) dados por las neuronas.

vectoresCostes = np.column_stack(
    (W[:,:,len(resultadosSimulacion.columns)-2].reshape(-1),
     W[:,:,len(resultadosSimulacion.columns)-1].reshape(-1)))

kMedias = KMeans(n_clusters=8).fit(vectoresCostes)


plt.figure()
for i, f in enumerate(resultadosSimulacion.columns):
    for j in np.arange(som._weights.shape[0]):
        for k in np.arange(som._weights.shape[1]):
            plt.subplot(3, 3, i+1)
            plt.title(f)
            muestra = np.asmatrix([[W[j, k, -2], W[j, k, -1]], ])
            plt.text(j, k, str(int(kMedias.predict(muestra))), fontdict=fuente)
            plt.pcolor(W[:, :, i].T, cmap='coolwarm')
            plt.xticks(np.arange(anchuraNeuronas+1))
            plt.yticks(np.arange(anchuraNeuronas+1))
plt.tight_layout(pad=0.5)

clusteres = list(enumerate(list(kMedias.cluster_centers_)))
clusteres.sort(key=lambda tupla: tupla[1][0] + tupla[1][1])


clusteresOrdenados = list(map(lambda tupla: tupla[0],clusteres))

# Más enreo áun: Señalar los valores pero solo para aquellas neuronas que están en el mejor cluster:
plt.figure()
for i, f in enumerate(resultadosSimulacion.columns):
    for j in np.arange(som._weights.shape[0]):
        for k in np.arange(som._weights.shape[1]):
            plt.subplot(3, 3, i+1)
            plt.title(f)
            muestra = np.asmatrix([[W[j, k, -2], W[j, k, -1]], ])
            cluster = int(kMedias.predict(muestra))

            if cluster == clusteresOrdenados[0]:
                plt.text(j, k, str(int(round(escala.inverse_transform(W)[j, k, i]))), fontdict=fuente)
                # plt.text(j, k, str(int(round(escala.inverse_transform(W[j])[k, i]))), fontdict=fuente)

            plt.pcolor(W[:, :, i].T, cmap='coolwarm')
            plt.xticks(np.arange(anchuraNeuronas+1))
            plt.yticks(np.arange(anchuraNeuronas+1))
plt.tight_layout(pad=0.5)



vectoresCostesOrdenados = list(np.column_stack(
    (vectoresCostes,
     np.arange(0, len(vectoresCostes)))))
vectoresCostesOrdenados.sort(key=lambda tupla: tupla[0]+tupla[1])
indice1DMejorNeurona = int(vectoresCostesOrdenados[0][2])
mejorNeurona = escala.inverse_transform(W)[int(indice1DMejorNeurona/anchuraNeuronas),indice1DMejorNeurona%anchuraNeuronas,:]
#
plt.show()



print("ok")



