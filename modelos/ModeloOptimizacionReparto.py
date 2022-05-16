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

class ModeloOptimizacionReparto(ModeloMioV4):


    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloV6"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo

        # Por si acaso, inicializamos el reparto de carga como nulo:
        self.repartoCarga = None

        self.numTotalDispositivos = 0
        for numDispNodo in matrizCluster:
            self.numTotalDispositivos = self.numTotalDispositivos + numDispNodo

    # Método que introduce una distribución estática con los valores de núcleos activos:
    def introducirRepartoCarga(self, individuo):

        M = self.numMaxSubpoblaciones
        self.repartoCarga = []
        for i in range(0, M):
            # Obtenemos el dispositivo y el nodo concreto:
            dispositivoSobreTotal = math.floor(individuo[i] * self.numTotalDispositivos)
            nodo, dispositivo = self.__obtenerIndicesNodoDisp(dispositivoSobreTotal)
            numNucleosActivos = \
                math.ceil(float(self.datos['P_' + str(nodo + 1) + '_' + str(dispositivo + 1)] * individuo[i + M]))

            numNucleosActivos = 1 if numNucleosActivos == 0 else numNucleosActivos

            # Guardamos la asignación en el reparto de carga:
            tupla = {"nodo": nodo, "dispositivo": dispositivo, "numNucleosActivos": numNucleosActivos}
            self.repartoCarga.append(tupla)

    # Método (back-end) que devuelve el reparto de carga y el núm. de núcleos activos que contiene el modelo:
    def obtenerIndividuoReparto(self):
        reparto = self.repartoCarga
        individuo = [None] * self.numMaxSubpoblaciones * 2

        i = 0
        for asignacion in reparto:
            nodo = asignacion["nodo"]
            dispositivo = asignacion["dispositivo"]
            numNucleosActivos = asignacion["numNucleosActivos"]

            # Averiguamos el número de núcleos totales del dispositivo:
            numNucleosTotales = self.datos['P_' + str(nodo+1) + '_' + str(dispositivo+1)]

            # Averiguamos qué dispositivo es sobre el total:
            dispositivoSobreTotal = 0
            for indiceNodo in range(0, len(self.matrizCluster)):
                if nodo == indiceNodo:
                    dispositivoSobreTotal = dispositivoSobreTotal + dispositivo
                    break
                else:
                    dispositivoSobreTotal = dispositivoSobreTotal + self.matrizCluster[indiceNodo]

            # Introducimos los valores para la correspondiente subpoblación en el individuo:
            individuo[i] = float(dispositivoSobreTotal / self.numTotalDispositivos)
            individuo[i + self.numMaxSubpoblaciones] = float(numNucleosActivos/numNucleosTotales)

            i = i+1

        return individuo


    # Método para buscar repartos de carga y configuraciones de núcleos activos óptimos:
    def ajustarParametros(self):
        algoritmoGenetico = NSGA2(self, modeloOptimizacionReparto=True)
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

    # Método (front-end) que devuelve el reparto de carga y el núm. de núcleos activos que contiene el modelo:
    def construirIndividuo(self):

        # Si no hay un reparto definido, se propone el reparto dinámico que se produciría:
        if (self.repartoCarga is None) or (self.repartoCarga == []):
            self.repartoCarga = self.calcularDistribucionCargaDinamica(self.numMaxSubpoblaciones)
            individuo = self.obtenerIndividuoReparto()
            self.introducirRepartoCarga(individuo)
        else:
            individuo = self.obtenerIndividuoReparto()

        return individuo

    # Función que devuelve los valores de fitness del modelo:
    def calcularFitness(self, genes, devolverResultados=False):
        numMaxSubpoblaciones = self.numMaxSubpoblaciones
        modelo = self
        datos = self.datos

        # Dejamos escrito en el modelo el reparto de la carga:
        self.introducirRepartoCarga(genes)

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
                                        (1 / numMaxSubpoblaciones) *(costes['T'] - tiempoReal))
                                    # signoTiempo * (1 / numMaxSubpoblaciones) * math.pow(costes['T'] - tiempoReal, 2))
            difCuadrMediaEnergia = float(difCuadrMediaEnergia + \
                                         (1 / numMaxSubpoblaciones) * (costes['energy'] - energiaReal))
                                    # signoEnergia * (1 / numMaxSubpoblaciones) * math.pow(costes['energy'] - energiaReal, 2))

        if not devolverResultados:
            return difCuadrMediaTiempo, difCuadrMediaEnergia
        else:
            return costesReales, costesComputados

    # Método que devuelve los índices de nodo y dispositivo dado un índice de dispositivo sobre total:
    def __obtenerIndicesNodoDisp(self, dispositivoSobreTotal):
        k = 0
        dispositivo = 0
        nodo = 0
        for indiceNodo in range(0, len(self.matrizCluster)):
            for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
                if k == dispositivoSobreTotal:
                    nodo = indiceNodo
                    dispositivo = indiceDisp
                k = k + 1
        return nodo, dispositivo

    # Método que devuelve si el individuo dado representa una solución consistente:
    def esSolucionConsistente(self,individuo):
        res = True
        M = self.numMaxSubpoblaciones

        for i in range(0,M):
            # Obtenemos el dispositivo y el nodo concreto:
            dispositivoSobreTotal = math.floor(individuo[i] * self.numTotalDispositivos)
            nodo, dispositivo = self.__obtenerIndicesNodoDisp(dispositivoSobreTotal)
            numNucleosActivos = individuo[i + M]


            if self.datos['P_' + str(nodo+1) + '_' + str(dispositivo+1)] < numNucleosActivos:
                res = False
                break

        return res

    # Método que calcula la distribución estática de carga del modelo:
    def __calcularDistribucionCarga(self, numSubpoblacionesTotales):

        # En esta versión se aplica la distribución de carga dada en el modelo:
        # Guardamos el historial de ejecución de cada nodo:
        historialEjecucion = []

        # Marcas temporales de los dispositivos:
        marcasTemporales = []
        tiemposEjecucion = []
        for indiceNodo in range(0, len(self.matrizCluster)):
            marcasTemporales.append([])
            tiemposEjecucion.append([])
            for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
                marcasTemporales[indiceNodo].append(0.0)
                tiemposEjecucion[indiceNodo].append(0.0)

        # Recorremos la distribución de carga dada. Por cada subpoblación añadimos tiempo consumido si corresponde.
        # OJO: No se imputan costes cuando nos hemos quedado sin subpoblaciones.
        numSubpoblacionesComputadas = 0
        marcaTemporalActual = 0.0
        for asignacion in self.repartoCarga:
            if numSubpoblacionesComputadas < numSubpoblacionesTotales:
                nodo = asignacion['nodo']
                dispositivo = asignacion['dispositivo']
                numNucleosActivos = asignacion['numNucleosActivos']

                # Calculamos el tiempo y la energía consumidos en la ejecución:
                tiempo = self.__calcularCosteTiempoDispositivo(numSubpoblacionesTotales, 1, nodo+1,
                                             dispositivo+1, numNucleosActivos=numNucleosActivos)
                energia = self.__calcularCosteEnergiaActivaDispositivo(tiempo, nodo+1,
                                             dispositivo+1, numNucleosActivos=numNucleosActivos)

                # Apuntamos la ejecución en el historial:
                historialEjecucion.append({
                    'nodo': nodo,
                    'dispositivo': dispositivo,
                    'numNucleosActivos': numNucleosActivos,
                    'tiempo': tiempo,
                    'energia': energia
                })

                # Si se ha despachado una subpoblación a un dispositivo cuya marca temporal de fin es superior a la actual,
                # actualizamos la marca temporal a la de dicho dispositivo (más el coste de envía de subpop).
                # Esto significa que se ha esperado a que termine el dispositivo para darle una nueva subpoblación
                if marcasTemporales[nodo][dispositivo] > marcaTemporalActual:
                    # Actualizamos la marca temporal actual:
                    marcaTemporalActual = marcasTemporales[nodo][dispositivo] + self.datos['Tcom']

                # De lo contrario, ponemos como marca temporal de inicio la marca actual:
                else:
                    # Actualizamos la marca temporal actual:
                    marcaTemporalActual = marcaTemporalActual + self.datos['Tcom']

                # La marca temporal de inicio dicho dispositivo se actualiza.
                marcasTemporales[nodo][dispositivo] = marcaTemporalActual
                historialEjecucion[-1]['inicio'] = marcaTemporalActual - self.datos['Tcom']

                # La marca temporal de dicho dispositivo se actualiza a su vez.
                marcasTemporales[nodo][dispositivo] = marcasTemporales[nodo][dispositivo] + tiempo + self.datos['Tcom']
                historialEjecucion[-1]['fin'] = marcasTemporales[nodo][dispositivo]


                tiemposEjecucion[nodo][dispositivo] = tiemposEjecucion[nodo][dispositivo] + tiempo

                numSubpoblacionesComputadas = numSubpoblacionesComputadas +1
            else:
                break

        # Calculamos el tiempo total consumido:
        self.tiempoTotal = max(historialEjecucion, key=lambda asignacion: asignacion['fin'])['fin']

        self.historialEjecucion = historialEjecucion

        self.tiemposEjecucion = tiemposEjecucion

    # Método para calcular la distribución dinámica de carga para la configuración y parámetros del modelo
    def calcularDistribucionCargaDinamica(self, numSubpoblacionesTotales):
        repartoCarga = []

        # En esta versión se aplica una distribución de carga que simula (en tiempo) aquella usada por el algoritmo V3:
        # las subpoblaciones se distribuyen por las solicitudes de los dispositivos.

        # Etapa = Momento en el que hay disponible una o más subpoblaciones sin computar para una o más dispositivos disponibles.
        numDispositivosTotales = 0
        costesTemporales = []
        rankingNodoDisp = []

        # Inicializamos las estructuras de datos que se hacen uso en la planificación:
        for indiceNodo in range(0, len(self.matrizCluster)):
            numDispositivosTotales = numDispositivosTotales + self.matrizCluster[indiceNodo]

            costesTemporales.append([])
            # Calculamos el coste temporal de una subpoblación en cada dispositivo:
            for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
                # Omitimos el dispositivo si tiene todos los núcleos deshabilitados:
                if self.datos['P_' + str(indiceNodo + 1) + '_' + str(indiceDisp + 1)] > 0:
                    costesTemporales[indiceNodo] \
                        .append(self.__calcularCosteTiempoDispositivo(numSubpoblacionesTotales, 1, indiceNodo + 1,
                                                                      indiceDisp + 1))
                    rankingNodoDisp.append([indiceNodo, indiceDisp, 0.0])
                # NUEVO (22/06/2021):
                else:  # Núm nucleos = 0 => Dispositivo apagado
                    costesTemporales[indiceNodo].append(0)

        # Repartimos las subpoblaciones hasta que no quede ninguna:
        numSubpoblacionesRestantes = numSubpoblacionesTotales
        while numSubpoblacionesRestantes > 0:
            # Escogemos el siguiente dispositivo a asignar una subpoblación
            indiceNodo = rankingNodoDisp[0][0]
            indiceDisp = rankingNodoDisp[0][1]

            # Actualizamos la m2arca temporal de fin de ejecución del dispositivo más el coste de comunicación:
            rankingNodoDisp[0][2] = rankingNodoDisp[0][2] + costesTemporales[indiceNodo][indiceDisp] + 2 * self.datos[
                'Tcom']

            # Guardamos la asignación en el reparto de carga:
            repartoCarga.append(
                {
                    'nodo': indiceNodo,
                    'dispositivo': indiceDisp,
                    'numNucleosActivos': int(self.datos['P_' + str(indiceNodo + 1) + '_' + str(indiceDisp + 1)])
                }
            )

            numSubpoblacionesRestantes = numSubpoblacionesRestantes - 1

            # Reordenamos el ranking:
            rankingNodoDisp.sort(key=lambda dispositivo: dispositivo[2])

        return repartoCarga

    # Método que devuelve los costes de tiempo y energía de un nodo para un núm. de subpoblaciones dado:
    def __calcularCosteTiempoEnergiaNodo(self, numSubpoblacionesTotales, indiceNodo, matrizCarga):
        res = dict()
        D = self.datos  # Para abreviar el código
        i = str(indiceNodo)

        # Calculamos la energía consumida de los dispositivos del nodo:

        # Energías activas:
        energiasActivasConsumidas = list(
            filter(lambda ejecucion: ejecucion['nodo'] == indiceNodo, self.historialEjecucion))
        energiasActivasConsumidas = list(map(lambda ejecucion: float(ejecucion['energia']), energiasActivasConsumidas))

        # Un apaño para evitar error por lista vacía:
        energiasActivasConsumidas.append(0.0)

        energiaActivaDispositivos = reduce(lambda e1, e2: e1+e2, energiasActivasConsumidas)

        # Energías idle/ociosas:
        # Calculamos la energía consumida de manera ociosa por cada dispositivo:
        energiaIdleDispositivos = 0.0
        D = self.datos
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            tiempoIdle = self.tiempoTotal - self.tiemposEjecucion[indiceNodo][indiceDisp]
            energiaIdle = D['Pow_' + str(indiceNodo+1) + '_' + str(indiceDisp+1) + '_idle'] * tiempoIdle
            energiaIdleDispositivos = energiaIdleDispositivos + energiaIdle

        energiaDispositivosTotal = energiaActivaDispositivos + energiaIdleDispositivos

        ''''''
        # Finalmente, calculamos la energía como la energía de los dispositivos en ejecución más
        # el consumo en estado ocioso durante el tiempo de overhead y comunicación:
        potenciaIdleTotal = 0.0
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            potenciaIdleTotal = potenciaIdleTotal + D[
                'Pow_' + str(indiceNodo + 1) + '_' + str(indiceDisp + 1) + '_idle']
        energiaNodo = energiaDispositivosTotal + (D['Tmaster'] + D['Tcom']) * potenciaIdleTotal

        return energiaNodo

    # Método que devuelve el coste temporal de la ejecución en un dispositivo:
    def __calcularCosteTiempoDispositivo(self, numSubpoblacionesTotales, numSubpoblaciones, indiceNodo,
                                         indiceDispostivo, numNucleosActivos=-1):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        numIndSubp = (D['N'] / numSubpoblacionesTotales) * (1 + D['tasaCruce'])

        # Si no se indica el núm. de nucleos activos, todos los núcleos están activos:
        if numNucleosActivos == -1:
            numNucleosActivos = D['P_' + i + '_' + j]

        if numNucleosActivos > 0:
            tiempo = D['gen'] * numSubpoblaciones * (
                        math.ceil(numIndSubp / numNucleosActivos) * D['W_' + i + '_' + j] / D['F_' + i + '_' + j])
        else:
            tiempo = 0

        return tiempo

    # Método que devuelve el coste energético activo de un dispositivo:
    def __calcularCosteEnergiaActivaDispositivo(self, tiempoDisp, indiceNodo, indiceDispostivo, numNucleosActivos=-1):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        # Si no se indica el núm. de nucleos activos, todos los núcleos están activos:
        if numNucleosActivos == -1:
            numNucleosActivos = D['P_' + i + '_' + j]

        energia = numNucleosActivos * D['Pow_' + i + '_' + j] * tiempoDisp
        return energia

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
