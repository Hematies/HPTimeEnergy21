import scipy.io as scipy
import numpy as np
import math
import random
from statistics import mean, stdev

from AG.NSGA2 import NSGA2
from modelos.ModeloV4 import  Modelo as ModeloMioV4
from modelos.ModeloV2 import  Modelo as ModeloMioV2
from copy import copy

from sklearn.preprocessing import minmax_scale, scale,MinMaxScaler, Normalizer


MODELAR_TIEMPO_ENERGIA = 0
MODELAR_TIEMPO = 1
MODELAR_ENERGIA = 2


# Clase que implementa el modelo V6:
class Modelo(ModeloMioV2):

    # Modelo V2, solo que primero se ajustan los parámetros relacionados con el coste temporal, y luego, separadamente,
    # aquellos parámetros relacionados con el coste energético.

    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloV6"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo
        self.MAX_VALOR_P = 10.0

        self.tipoModelado = MODELAR_TIEMPO_ENERGIA

        # PARA DEPURAR:
        self.tiemposRegresion = dict()

    # Método para ajustar los parámetros del modelo:
    def ajustarParametros(self):
        algoritmoGenetico = NSGA2(self)


        # Modelamos el tiempo:
        self.tipoModelado = MODELAR_TIEMPO
        poblacionResultante, stats = algoritmoGenetico.calcularNSGA2()
        print('==================')
        poblacionResultante.sort(key=lambda x: x.fitness.values[0])
        mejorIndividuo = poblacionResultante[0]
        self.introducirParametros(mejorIndividuo, esGen=True)

        # Modelamos la energía:
        self.tipoModelado = MODELAR_ENERGIA
        poblacionResultante, stats = algoritmoGenetico.calcularNSGA2()
        print('==================')
        poblacionResultante.sort(key=lambda x: x.fitness.values[1])
        mejorIndividuo = poblacionResultante[0]
        self.introducirParametros(mejorIndividuo, esGen=True)

        # Modelamos ambos tiempo y energía:
        self.tipoModelado = MODELAR_TIEMPO_ENERGIA
        poblacionResultante, stats = algoritmoGenetico.calcularNSGA2(NGEN=1, MU=12)
        print('==================')
        mejorIndividuo = None
        mejorFitness = None
        listaFitness = list(map(lambda ind: ind.fitness.values, poblacionResultante))

        listaFitnessTiempo = list(map(lambda fitness: fitness[0], listaFitness))
        listaFitnessEnergia = list(map(lambda fitness: fitness[1], listaFitness))


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

        return mejorIndividuo, mejorFitness

    # Método para devolver los valores (reescalados) de los parámetros originalmente guardados en el workspace dado:
    def construirIndividuoOriginal(self):
        datos = copy(self.datosOriginales)
        parametros = []

        for indice in range(1, 4):
            i = str(indice)
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
                parametros.append(datos['Wgpu' + i] / self.MAX_VALOR_W)
                parametros.append(datos['Wcpu' + i] / self.MAX_VALOR_W)
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
                parametros.append(datos['POW_gpu' + i] / self.MAX_VALOR_P)
                parametros.append(datos['POW_cpu' + i] / self.MAX_VALOR_P)
                parametros.append(datos['POW_gpu' + i + '_idle'] / self.MAX_VALOR_P)
                parametros.append(datos['POW_cpu' + i + '_idle'] / self.MAX_VALOR_P)

        # Otros parámetros a ajustar:
        if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
            parametros.append(datos['Tcom'] / self.MAX_VALOR_T)
            parametros.append(datos['Tmaster'] / self.MAX_VALOR_T)
        if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
            parametros.append(datos['POW_cpu0'] / self.MAX_VALOR_P)
            parametros.append(datos['POW_sw'] / self.MAX_VALOR_P)
        return parametros

    # Método que devuelve los valores (reescalados) de los parámetros guardados en el workspace dado:
    def construirIndividuo(self):
        datos = copy(self.datos)
        parametros = []

        for indice in range(1, 4):
            i = str(indice)
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
                parametros.append(datos['Wgpu' + i] / self.MAX_VALOR_W)
                parametros.append(datos['Wcpu' + i] / self.MAX_VALOR_W)
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
                parametros.append(datos['POW_gpu' + i] / self.MAX_VALOR_P)
                parametros.append(datos['POW_cpu' + i] / self.MAX_VALOR_P)
                parametros.append(datos['POW_gpu' + i + '_idle'] / self.MAX_VALOR_P)
                parametros.append(datos['POW_cpu' + i + '_idle'] / self.MAX_VALOR_P)

        # Otros parámetros a ajustar:
        if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
            parametros.append(datos['Tcom'] / self.MAX_VALOR_T)
            parametros.append(datos['Tmaster'] / self.MAX_VALOR_T)
        if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
            parametros.append(datos['POW_cpu0'] / self.MAX_VALOR_P)
            parametros.append(datos['POW_sw'] / self.MAX_VALOR_P)
        return parametros

    # Método que introduce los parámetros dados como parámetros del modelo:
    def introducirParametros(self, parametros, esGen=False):
        if not esGen:
            for indice in range(1, 4):
                i = str(indice)
                if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
                    self.datos['Wgpu' + i] = parametros['Wgpu' + i]
                    self.datos['Wcpu' + i] = parametros['Wcpu' + i]

                if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
                    self.datos['POW_gpu' + i] = parametros['POW_gpu' + i]
                    self.datos['POW_cpu' + i] = parametros['POW_cpu' + i]
                    self.datos['POW_gpu' + i + '_idle'] = parametros['POW_gpu' + i + '_idle']
                    self.datos['POW_cpu' + i + '_idle'] = parametros['POW_cpu' + i + '_idle']

            # Otros parámetros a ajustar:
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
                self.datos['Tcom'] = parametros['Tcom']
                self.datos['Tmaster'] = parametros['Tmaster']
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
                self.datos['POW_cpu0'] = parametros['POW_cpu0']
                self.datos['POW_sw'] = parametros['POW_sw']
        else:
            j = 0
            for indice in range(1, 4):
                i = str(indice)
                if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
                    self.datos['Wgpu' + i] = int(parametros[j] * self.MAX_VALOR_W)
                    self.datos['Wcpu' + i] = int(parametros[j + 1] * self.MAX_VALOR_W)
                    j = j+2
                if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
                    self.datos['POW_gpu' + i] = parametros[j] * self.MAX_VALOR_P
                    self.datos['POW_cpu' + i] = parametros[j + 1] * self.MAX_VALOR_P
                    self.datos['POW_gpu' + i + '_idle'] = parametros[j + 2] * self.MAX_VALOR_P
                    self.datos['POW_cpu' + i + '_idle'] = parametros[j + 3] * self.MAX_VALOR_P
                    j = j + 4

            # Otros parámetros a ajustar:
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_TIEMPO:
                self.datos['Tcom'] = parametros[j] * self.MAX_VALOR_T
                self.datos['Tmaster'] = parametros[j + 1] * self.MAX_VALOR_T
                j = j + 2
            if self.tipoModelado == MODELAR_TIEMPO_ENERGIA or self.tipoModelado == MODELAR_ENERGIA:
                self.datos['POW_cpu0'] = parametros[j] * self.MAX_VALOR_P
                self.datos['POW_sw'] = parametros[j + 1] * self.MAX_VALOR_P
                j = j + 2

        # Traducimos los nombres de las variables para que sean acordes a los modelos míos:
        self.__traducirVariablesWorkspace()

    # Método que devuelve los valores de fitness del modelo:
    def calcularFitness(self, genes, devolverResultados=False, usarDatosModelo=False):
        numMaxSubpoblaciones = self.numMaxSubpoblaciones
        modelo = self
        datos = self.datos

        if not usarDatosModelo:

            modelo.introducirParametros(genes, esGen=True)

        # Recogemos los costes reales:
        tiemposReales = datos["time_v1v2v3_all"][:, 2]
        energiasReales = datos["energ_v1v2v3_all"][:, 2]

        errorCuadrMedioTiempo = 0.0
        errorCuadrMedioEnergia = 0.0

        costesComputados = []
        costesReales = []

        for numSubpoblaciones in range(1, numMaxSubpoblaciones + 1):
            # Calculamos los costes computados
            costes = modelo.computarCosteTiempoEnergia(numSubpoblaciones, usarDatosModelo=usarDatosModelo)
            costesComputados.append(costes)

            # Recogemos los costes reales para el núm de subp. dado:
            tiempoReal = tiemposReales[numSubpoblaciones - 1] / 1000  # Está en milisegudos
            # tiempoReal = tiemposReales[numSubpoblaciones - 1] # Está en segundos
            energiaReal = energiasReales[numSubpoblaciones - 1]
            c = dict()
            c['T'] = tiempoReal
            c['energy'] = energiaReal
            costesReales.append(c)

            errorCuadrMedioTiempo = float(errorCuadrMedioTiempo + \
                                    (1 / numMaxSubpoblaciones) * math.pow(costes['T'] - tiempoReal, 2))
            errorCuadrMedioEnergia = float(errorCuadrMedioEnergia + \
                                     (1 / numMaxSubpoblaciones) * math.pow(costes['energy'] - energiaReal, 2))

        errorCuadrMedioTiempo = math.sqrt(errorCuadrMedioTiempo)
        errorCuadrMedioEnergia = math.sqrt(errorCuadrMedioEnergia)

        # Un truco para devolver solo un error de los dos tipos:
        if not self.tipoModelado == MODELAR_TIEMPO_ENERGIA:
            if self.tipoModelado == MODELAR_TIEMPO:
                # return errorCuadrMedioTiempo, 1.0
                errorCuadrMedioEnergia = 1.0
            elif self.tipoModelado == MODELAR_ENERGIA:
                # return 1.0, errorCuadrMedioEnergia
                errorCuadrMedioTiempo = 1.0


        if not devolverResultados:
            return errorCuadrMedioTiempo, errorCuadrMedioEnergia
        else:
            return costesReales, costesComputados

    # Método que define la consistencia de los valores de los parámetros de un individuo:
    def esSolucionConsistente(self,genes):
        res = True

        if self.tipoModelado == MODELAR_TIEMPO_ENERGIA:
            for i in range(0, len(self.matrizCluster)):
                # Las potencias idle no pueden ser mayores que las activas:
                inconsistenciaGPU = genes[i*6 + 2] < genes[i*6 + 4]
                inconsistenciaCPU = genes[i*6 + 3] < genes[i*6 + 5]

                if inconsistenciaCPU or  inconsistenciaGPU:
                    res = False
        elif self.tipoModelado == MODELAR_ENERGIA:
            for i in range(0, len(self.matrizCluster)):
                # Las potencias idle no pueden ser mayores que las activas:
                inconsistenciaGPU = genes[i * 4] < genes[i * 4 + 2]
                inconsistenciaCPU = genes[i * 4 + 1] < genes[i * 4 + 3]

                if inconsistenciaCPU or inconsistenciaGPU:
                    res = False
        return res

