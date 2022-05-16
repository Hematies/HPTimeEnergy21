import scipy.io as scipy
import numpy as np
import math
import random
from statistics import mean, stdev

from AG.NSGA2 import NSGA2
from modelos.ModeloV4 import  Modelo as ModeloMioV4
from modelos.ModeloV2 import  Modelo as ModeloMioV5
from copy import copy
from functools import reduce
from sklearn.preprocessing import minmax_scale, scale,MinMaxScaler

from sklearn.metrics import r2_score

import pprint

# Clase que implementa el modelo de optimización por núcleos activos para todo núm. de subpoblación.
class ModeloOptimizacionNucleosActivos(ModeloMioV4):

    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloOptimizacionNucleosActivos"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo

        self.numTotalDispositivos = 0
        for numDispNodo in matrizCluster:
            self.numTotalDispositivos = self.numTotalDispositivos + numDispNodo

        self.nucleosTotales = []
        self.__traducirVariablesWorkspace()
        for i in range(0, len(self.matrizCluster)):
            self.nucleosTotales.append([])
            for j in range(0, self.matrizCluster[i]):
                self.nucleosTotales[i].append(int(self.datos['P_' + str(i + 1) + '_' + str(j + 1)]))
                # if not (i==0 and j==0):
                if not (i == 0 and j == 1):
                    self.datos['P_' + str(i + 1) + '_' + str(j + 1)] = 0

    # Método que devuelve si la solución dada por los genes es consistente:
    def esSolucionConsistente(self,genes):
        hayAlgunNucleoActivo = False
        k = 0
        # print(individuo)
        for i in range(0, len(self.matrizCluster)):
            for j in range(0, self.matrizCluster[i]):
                hayAlgunNucleoActivo = hayAlgunNucleoActivo or (round(genes[k] * self.nucleosTotales[i][j]) > 0)
                k = k + 1

        return not hayAlgunNucleoActivo


    # Función que traduce los nombres de las variables del workspace a una nueva nomenclatura:
    def __traducirVariablesWorkspace(self):
        # Hay dos nodos con CPU y GPU
        for i in range(1, 4):
            for j in range(1, 3):
                self.datos['P_' + str(i) + '_' + str(j)] = self.datos['Pgpu' + str(i)] if j == 1 \
                    else self.datos['Pcpu' + str(i)]
                self.datos['W_' + str(i) + '_' + str(j)] = self.datos['Wgpu' + str(i)] if j == 1 \
                    else self.datos['Wcpu' + str(i)]
                self.datos['F_' + str(i) + '_' + str(j)] = self.datos['Fgpu' + str(i)] if j == 1 \
                    else self.datos['Fcpu' + str(i)]
                self.datos['Pow_' + str(i) + '_' + str(j)] = self.datos['POW_gpu' + str(i)] if j == 1 \
                    else self.datos['POW_cpu' + str(i)]
                self.datos['Pow_' + str(i) + '_' + str(j) + '_idle'] = self.datos[
                    'POW_gpu' + str(i) + '_idle'] if j == 1 \
                    else self.datos['POW_cpu' + str(i) + '_idle']

    # Método que introduce una configuración de números reescalados de núcleos activos en el modelo:
    def introducirNucleosActivos(self, individuo):
        k = 0
        # print(individuo)
        for i in range(0, len(self.matrizCluster)):
            for j in range(0, self.matrizCluster[i]):
                self.datos['P_' + str(i+1) + '_' + str(j+1)] = round(individuo[k] * self.nucleosTotales[i][j])

                k = k + 1

    # Método para buscar la configuración óptima de núcleos activos:
    def ajustarParametros(self):
        algoritmoGenetico = NSGA2(self, modeloOptimizacionNucleosActivos=True)
        poblacionResultante, stats = algoritmoGenetico.calcularNSGA2()
        print('==================')

        mejorIndividuo = None
        mejorFitness = None
        listaFitness = list(map(lambda ind: ind.fitness.values, poblacionResultante))


        escala = MinMaxScaler()
        escala = escala.fit(listaFitness)

        listaFitness = escala.transform(listaFitness)

        i = 0
        for individuo in poblacionResultante:
            # Así se obtiene una medida normalizada: dividiendo por el mínimo valor en cada caso
            v = math.pow(listaFitness[i][0], 2) + math.pow(listaFitness[i][1], 2)

            if ((mejorFitness == None) or (mejorFitness >= v)):
                mejorIndividuo = individuo
                mejorFitness = v
            i = i+1

        return mejorIndividuo, mejorIndividuo.fitness

    # Método (back-end) que devuelve la configuración (reescalada) de núcleos activos del modelo:
    def obtenerIndividuoNucleosActivos(self):
        individuo = []
        k = 0
        for i in range(0, len(self.matrizCluster)):
            for j in range(0, self.matrizCluster[i]):
                individuo.append(float(self.datos['P_' + str(i + 1) + '_' + str(j + 1)] / self.nucleosTotales[i][j]))
                k = k + 1
        return individuo

    # Método (front-end) que devuelve la configuración (reescalada) de núcleos activos del modelo:
    def construirIndividuo(self):
        individuo = []
        if self.nucleosTotales == None or self.nucleosTotales == []:
            self.nucleosTotales = []
            self.__traducirVariablesWorkspace()
            for i in range(0, len(self.matrizCluster)):
                self.nucleosTotales.append([])
                for j in range(0, self.matrizCluster[i]):
                    self.nucleosTotales[i].append(int(self.datos['P_' + str(i + 1) + '_' + str(j + 1)]))
                    if not (i == 0 and j == 1):
                        self.datos['P_' + str(i + 1) + '_' + str(j + 1)] = 0
            individuo = self.obtenerIndividuoNucleosActivos()
        else:
            individuo = self.obtenerIndividuoNucleosActivos()
        return individuo

    # Función que devuelve los valores de fitness del modelo:
    def calcularFitness(self, genes, devolverResultados=False):
        numMaxSubpoblaciones = self.numMaxSubpoblaciones
        modelo = self
        datos = self.datos


        # Dejamos escrito en el modelo el reparto de la carga:
        self.introducirNucleosActivos(genes)

        # Recogemos los costes reales:
        tiemposReales = datos["time_v1v2v3_all"][:, 2]
        energiasReales = datos["energ_v1v2v3_all"][:, 2]

        difCuadrMediaTiempo = 0.0
        difCuadrMediaEnergia = 0.0

        costesComputados = []
        costesReales = []

        for numSubpoblaciones in range(1, numMaxSubpoblaciones + 1):
            # Calculamos los costes computados
            costes = modelo.computarCosteTiempoEnergia(numSubpoblaciones)
            costesComputados.append(costes)

            # Recogemos los costes reales para el núm de subp. dado:
            tiempoReal = tiemposReales[numSubpoblaciones - 1] / 1000  # Está en milisegudos
            energiaReal = energiasReales[numSubpoblaciones - 1]
            c = dict()
            c['T'] = tiempoReal
            c['energy'] = energiaReal
            costesReales.append(c)

            # Devolveremos las diferencias cuadráticas medias, conservando el signo de las diferencias:

            difCuadrMediaTiempo = float(difCuadrMediaTiempo + \
                                        (1 / numMaxSubpoblaciones) * (costes['T'] - tiempoReal))
            difCuadrMediaEnergia = float(difCuadrMediaEnergia + \
                                         (1 / numMaxSubpoblaciones) * (costes['energy'] - energiaReal))


        if not devolverResultados:
            return difCuadrMediaTiempo, difCuadrMediaEnergia
        else:
            return costesReales, costesComputados

    # Método que devuelve los costes de tiempo y energía para un núm. de subpoblaciones dado:
    def computarCosteTiempoEnergia(self, numSubpoblacionesTotales, usarDatosModelo=False):
        res = dict()

        D = self.datos  # Para abreviar el código

        # Calculamos la distribución de carga:
        matrizCarga = self.__calcularDistribucionCarga(numSubpoblacionesTotales)
        costeEnergiaNodos = []

        # Recopilamos los costes de los nodos:
        energiaTotalNodos = 0.0
        for indiceNodo in range(0, len(self.matrizCluster)):
            costeEnergiaNodos.append(
                self.__calcularCosteTiempoEnergiaNodo(numSubpoblacionesTotales, indiceNodo, matrizCarga))

            # Apuntamos la energia total de los nodos:
            energiaTotalNodos = energiaTotalNodos + costeEnergiaNodos[indiceNodo]

        # Calculamos el coste temporal del cluster:
        res['T'] = D['NGmig']*(D['Tmaster'] + self.tiempoTotal + D['Tcom'])

        # Calculamos el coste energético del cluster:
        res['energy'] = D['NGmig'] * (D['POW_cpu0'] * D['Tmaster'] + energiaTotalNodos + D['POW_sw'] * D['Tcom'])

        return res

    # Función que devuelve los costes energéticos del dispositivo para el tiempo de ejecución dado:
    def __calcularCosteEnergiaDispositivo(self, tiempoDisp, indiceNodo, indiceDispostivo):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        tiempoIdle = self.tiempoTotal - tiempoDisp
        energia = D['P_' + i + '_' + j] * D['Pow_' + i + '_' + j] * tiempoDisp + D['Pow_' + i + '_' + j + '_idle'] * tiempoIdle
        return energia

    # Método que computa la distribución de carga dinámica para el núm. de subpoblaciones dado:
    def __calcularDistribucionCarga(self, numSubpoblacionesTotales):
        res = []

        # En esta versión se aplica una distribución de carga que simula (en tiempo) aquella usada por el algoritmo V3:
        # las subpoblaciones se distribuyen por las solicitudes de los dispositivos.

        # Etapa = Momento en el que hay disponible una o más subpoblaciones sin computar para una o más dispositivos disponibles.
        numDispositivosTotales = 0
        costesTemporales = []
        rankingNodoDisp = []

        # Inicializamos las estructuras de datos que se hacen uso en la planificación:
        for indiceNodo in range(0, len(self.matrizCluster)):
            numDispositivosTotales = numDispositivosTotales + self.matrizCluster[indiceNodo]
            res.append([])

            costesTemporales.append([])
            # Calculamos el coste temporal de una subpoblación en cada dispositivo:
            for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
                res[indiceNodo].append(0)
                # Omitimos el dispositivo si tiene todos los núcleos deshabilitados:
                if self.datos['P_'+str(indiceNodo+1)+'_'+str(indiceDisp+1)] > 0:
                    costesTemporales[indiceNodo]\
                        .append(self.__calcularCosteTiempoDispositivo(numSubpoblacionesTotales, 1, indiceNodo+1, indiceDisp+1))
                    rankingNodoDisp.append([indiceNodo, indiceDisp, 0.0])
                else: # Núm nucleos = 0 => Dispositivo apagado
                    costesTemporales[indiceNodo].append(0)

        # Repartimos las subpoblaciones hasta que no quede ninguna:
        numSubpoblacionesRestantes = numSubpoblacionesTotales
        while numSubpoblacionesRestantes > 0:
            # Escogemos el siguiente dispositivo a asignar una subpoblación
            indiceNodo = rankingNodoDisp[0][0]
            indiceDisp = rankingNodoDisp[0][1]

            # Actualizamos la m2arca temporal de fin de ejecución del dispositivo más el coste de comunicación:
            rankingNodoDisp[0][2] = rankingNodoDisp[0][2] + costesTemporales[indiceNodo][indiceDisp] + 2 * self.datos['Tcom']

            # Actualizamos los números de subpoblaciones:
            res[indiceNodo][indiceDisp] = res[indiceNodo][indiceDisp] + 1
            numSubpoblacionesRestantes = numSubpoblacionesRestantes - 1

            # Reordenamos el ranking:
            rankingNodoDisp.sort(key=lambda dispositivo: dispositivo[2])

        self.tiempoTotal = max(rankingNodoDisp, key= lambda disp: disp[2])[2]
        return res

    # Función para calcular el coste temporal del dispositivo indicado:
    def __calcularCosteTiempoDispositivo(self, numSubpoblacionesTotales, numSubpoblaciones, indiceNodo, indiceDispostivo):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        numIndSubp = (D['N'] / numSubpoblacionesTotales) * (1 + D['tasaCruce'])


        if D['P_'+i+'_'+j] > 0:
            tiempo = D['gen'] * numSubpoblaciones * (math.ceil(numIndSubp/D['P_'+i+'_'+j]) * D['W_'+i+'_'+j] / D['F_'+i+'_'+j])
        else:
            tiempo = 0

        return tiempo

    # Función para calcular los costes de un nodo indicado:
    def __calcularCosteTiempoEnergiaNodo(self, numSubpoblacionesTotales,indiceNodo, matrizCarga):
        res = dict()
        D = self.datos  # Para abreviar el código
        i = str(indiceNodo)

        tiemposDispositivos = []

        # Calculamos el tiempo de ejecución de los dispositivos del nodo:
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            numSubpoblaciones = matrizCarga[indiceNodo][indiceDisp]
            costeTemporalDisp = \
                self.__calcularCosteTiempoDispositivo(numSubpoblacionesTotales, numSubpoblaciones, indiceNodo+1, indiceDisp+1)
            tiemposDispositivos.append(costeTemporalDisp)

        # Calculamos la energía consumida de los dispositivos del nodo:
        energiaDispositivosTotal = 0.0
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            energiaDispositivosTotal = energiaDispositivosTotal \
                + self.__calcularCosteEnergiaDispositivo(tiemposDispositivos[indiceDisp],
                                                         indiceNodo+1, indiceDisp+1)

        # Finalmente, calculamos la energía como la energía de los dispositivos en ejecución más
        # el consumo en estado ocioso durante el tiempo de overhead y comunicación:
        potenciaIdleTotal = 0.0
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            potenciaIdleTotal = potenciaIdleTotal + D['Pow_'+str(indiceNodo+1)+'_'+str(indiceDisp+1)+'_idle']
        energiaNodo = energiaDispositivosTotal + (D['Tmaster']+D['Tcom']) * potenciaIdleTotal

        return energiaNodo
