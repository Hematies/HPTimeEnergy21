import scipy.io as scipy
from numpy import number
import math
from modelos.ModeloV1 import  Modelo as ModeloMioV1

import pandas as pd

# Esta clase implementa el modelo V2
class Modelo(ModeloMioV1):

    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloV2"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo

    # Método para computar la distribución dinámica de carga del modelo:
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
                # NUEVO (22/06/2021):
                else: # Núm nucleos = 0 => Dispositivo apagado
                    costesTemporales[indiceNodo].append(0)

        # Repartimos las subpoblaciones hasta que no quede ninguna:
        numSubpoblacionesRestantes = numSubpoblacionesTotales
        while numSubpoblacionesRestantes > 0:
            # Escogemos el siguiente dispositivo a asignar una subpoblación
            indiceNodo = rankingNodoDisp[0][0]
            indiceDisp = rankingNodoDisp[0][1]

            # Actualizamos la m2arca temporal de fin de ejecución del dispositivo más el coste de comunicación:
            rankingNodoDisp[0][2] = rankingNodoDisp[0][2] + costesTemporales[indiceNodo][indiceDisp] \
                                    + 2 * self.datos['Tcom']

            # Actualizamos los números de subpoblaciones:
            res[indiceNodo][indiceDisp] = res[indiceNodo][indiceDisp] + 1
            numSubpoblacionesRestantes = numSubpoblacionesRestantes - 1

            # Reordenamos el ranking:
            rankingNodoDisp.sort(key=lambda dispositivo: dispositivo[2])

        self.tiempoTotal = max(rankingNodoDisp, key= lambda disp: disp[2])[2]
        return res

    # Método para calcular el coste de energía consumida del dispositivo para el tiempo activo dado:
    def __calcularCosteEnergiaDispositivo(self, tiempoDisp, indiceNodo, indiceDispostivo):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        tiempoIdle = self.tiempoTotal - tiempoDisp
        energia = D['Pow_'+i+'_'+j] * tiempoDisp + D['Pow_'+i+'_'+j+'_idle'] * tiempoIdle
        return energia

    # Método para calcular el coste tiempo del dispositivo para el número de subpoblaciones dado:
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
            # PARA DEPURAR
            # print('Num subp: ', numSubpoblacionesTotales)
            self.tiemposRegresion[','.join(str(x) for x in [indiceNodo, indiceDisp, numSubpoblacionesTotales])] = \
                [float(tiemposDispositivos[indiceDisp]), float(self.tiempoTotal-tiemposDispositivos[indiceDisp])]

            energiaDispositivosTotal = energiaDispositivosTotal \
                + self.__calcularCosteEnergiaDispositivo(tiemposDispositivos[indiceDisp],
                                                         indiceNodo+1, indiceDisp+1)

        ''''''
        # Finalmente, calculamos la energía como la energía de los dispositivos en ejecución más
        # el consumo en estado ocioso durante el tiempo de overhead y comunicación:
        potenciaIdleTotal = 0.0
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            potenciaIdleTotal = potenciaIdleTotal + D['Pow_'+str(indiceNodo+1)+'_'+str(indiceDisp+1)+'_idle']

        # En verdad hay que sumar las energías idle para dichos tiempos master y com ya que dichos tiempos son luego incluidos
        # en la función calcularCostesTiempoEnergia():
        energiaNodo = energiaDispositivosTotal + (D['Tmaster']+D['Tcom']) * potenciaIdleTotal


        # energiaNodo = energiaDispositivosTotal # OJO CON ESTE CAMBIO (10/08/21)

        return energiaNodo

    # Método para computar el coste en tiempo y energía del modelo para el núm. de subpoblaciones dado:
    def computarCosteTiempoEnergia(self, numSubpoblacionesTotales, usarDatosModelo=False):
        res = dict()

        if not usarDatosModelo:
            # Traducimos las variables del workspace de MATLAB a las variables aquí modeladas:
            self.__traducirVariablesWorkspace()
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
        res['T'] = float(D['NGmig']*(D['Tmaster'] + self.tiempoTotal + D['Tcom']))

        # Calculamos el coste energético del cluster:
        res['energy'] = float(D['NGmig'] * (D['POW_cpu0'] * D['Tmaster'] + energiaTotalNodos + D['POW_sw'] * D['Tcom']))


        return res
