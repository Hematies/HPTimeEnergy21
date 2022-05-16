import scipy.io as scipy
from numpy import number
import math
from modelos.ModeloV2 import  Modelo as ModeloMioV2


# Clase que implementa el modelo matemático V4:
class Modelo(ModeloMioV2):

    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloV4"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo
        self.MAX_VALOR_P = 10.0

    # Método para calcular el coste energético del dispositivo para el tiempo activo dado:
    def __calcularCosteEnergiaDispositivo(self, tiempoDisp, indiceNodo, indiceDispostivo):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        tiempoIdle = self.tiempoTotal - tiempoDisp
        energia = D['P_' + i + '_' + j] * D['Pow_' + i + '_' + j] * tiempoDisp + D['Pow_' + i + '_' + j + '_idle'] * tiempoIdle
        return energia

    # Método para calcular los costes en tiempo y energía para el núm. de subpoblaciones dado:
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
        res['T'] = D['NGmig']*(D['Tmaster'] + self.tiempoTotal + D['Tcom'])

        # Calculamos el coste energético del cluster:
        res['energy'] = D['NGmig'] * (D['POW_cpu0'] * D['Tmaster'] + energiaTotalNodos + D['POW_sw'] * D['Tcom'])

        return res

    # Método para introducir valores de parámetros en el modelo:
    def introducirParametros(self, parametros, esGen=False, datosPotenciaDisp=False):

        # Si los datos de potencia no son a nivel de cada núcleo, aproximamos la potencia mediante una división:
        if datosPotenciaDisp:
            if not esGen:
                for indice in range(1, 4):
                    i = str(indice)
                    parametros['POW_gpu' + i] = parametros['POW_gpu' + i] / self.datos['Pgpu'+i]
                    parametros['POW_cpu' + i] = parametros['POW_cpu' + i] / self.datos['Pcpu'+i]
            else:
                j = 0
                for indice in range(1, 4):
                    i = str(indice)
                    parametros[j + 2] = parametros[j + 2] / self.datos['Pgpu'+i]
                    parametros[j + 3] = parametros[j + 3] / self.datos['Pcpu'+i]
                    j = j + 6

        if not esGen:
            for indice in range(1, 4):
                i = str(indice)
                self.datos['Wgpu' + i] = parametros['Wgpu' + i]
                self.datos['Wcpu' + i] = parametros['Wcpu' + i]
                self.datos['POW_gpu' + i] = parametros['POW_gpu' + i]
                self.datos['POW_cpu' + i] = parametros['POW_cpu' + i]
                self.datos['POW_gpu' + i + '_idle'] = parametros['POW_gpu' + i + '_idle']
                self.datos['POW_cpu' + i + '_idle'] = parametros['POW_cpu' + i + '_idle']

            # Otros parámetros a ajustar:
            self.datos['Tcom'] = parametros['Tcom']
            self.datos['Tmaster'] = parametros['Tmaster']
            self.datos['POW_cpu0'] = parametros['POW_cpu0']
            self.datos['POW_sw'] = parametros['POW_sw']

        else:
            j = 0
            for indice in range(1, 4):
                i = str(indice)
                self.datos['Wgpu' + i] = int(parametros[j] * self.MAX_VALOR_W)
                self.datos['Wcpu' + i] = int(parametros[j + 1] * self.MAX_VALOR_W)
                self.datos['POW_gpu' + i] = parametros[j + 2] * self.MAX_VALOR_P
                self.datos['POW_cpu' + i] = parametros[j + 3] * self.MAX_VALOR_P
                self.datos['POW_gpu' + i + '_idle'] = parametros[j + 4] * self.MAX_VALOR_P
                self.datos['POW_cpu' + i + '_idle'] = parametros[j + 5] * self.MAX_VALOR_P
                j = j + 6

            # Otros parámetros a ajustar:
            self.datos['Tcom'] = parametros[j] * self.MAX_VALOR_T
            self.datos['Tmaster'] = parametros[j + 1] * self.MAX_VALOR_T
            self.datos['POW_cpu0'] = parametros[j + 2] * self.MAX_VALOR_P
            self.datos['POW_sw'] = parametros[j + 3] * self.MAX_VALOR_P

        # Por probar:
        self.__traducirVariablesWorkspace()