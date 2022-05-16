import scipy.io as scipy
import numpy as np
import math
import random
from statistics import mean, stdev

from AG.NSGA2 import NSGA2
from modelos.ModeloV6 import  Modelo as ModeloMioV6
from modelos.ModeloV2 import  Modelo as ModeloMioV2
from copy import copy
from PSO.PSO import PSO

from sklearn.preprocessing import minmax_scale, scale,MinMaxScaler


MODELAR_TIEMPO_ENERGIA = 0
MODELAR_TIEMPO = 1
MODELAR_ENERGIA = 2

# Clase que implementa el modelo V8:
class Modelo(ModeloMioV6):

    # Modelo V6 pero con PSO:

    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloV6"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo
        self.MAX_VALOR_P = 10.0

        self.tipoModelado = MODELAR_TIEMPO_ENERGIA

    # Método que evalúa la consistencia de los valores de los parámetros dados como "genes":
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

        # BORRAR LUEGO:
        res = True

        return res

    # Función que devuelve los valores de fitness del modelo de cada partícula de un enjambre:
    def calcularFitness(self, enjambre, devolverResultados=False, usarDatosModelo=False, PSO=False):
        if PSO:
            # Coleccionamos los costes de fitness de las partículas del enjambre:
            listaFitness = []
            for particula in enjambre:
                if self.tipoModelado == MODELAR_TIEMPO:
                    listaFitness.append(self.__calcularFitness(particula, devolverResultados=False, usarDatosModelo=False)[0])
                else: # elif self.tipoModelado == MODELAR_ENERGIA:
                    # Comprobamos primera la consistencia:
                    esConsistente = self.esSolucionConsistente(particula)

                    if esConsistente:
                        listaFitness.append(self.__calcularFitness(particula, devolverResultados=False, usarDatosModelo=False)[1])

                    else:
                        listaFitness.append(10e3)

            return np.array(listaFitness)
        else:
            return self.__calcularFitness(enjambre, devolverResultados=devolverResultados, usarDatosModelo=usarDatosModelo)


    # Función que devuelve los valores de fitness del modelo para una partícula
    def __calcularFitness(self, genes, devolverResultados=False, usarDatosModelo=False):
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

    # Método que ajusta los parámetros de un modelo:
    def ajustarParametros(self):
        mejorFitness = []
        mejorIndividuo = []

        # Modelamos el tiempo:
        self.tipoModelado = MODELAR_TIEMPO
        algoritmoParticulas = PSO(self)
        individuoResultante = algoritmoParticulas.calcularPSO()
        print('==================')
        self.introducirParametros(individuoResultante, esGen=True)
        # mejorFitness.append(self.__calcularFitness(individuoResultante))

        # Modelamos la energía:
        self.tipoModelado = MODELAR_ENERGIA
        algoritmoParticulas = PSO(self)
        individuoResultante = algoritmoParticulas.calcularPSO()
        print('==================')
        self.introducirParametros(individuoResultante, esGen=True)
        # mejorFitness.append(self.__calcularFitness(individuoResultante))

        # Modelamos ambos tiempo y energía:
        self.tipoModelado = MODELAR_TIEMPO_ENERGIA

        mejorIndividuo = self.construirIndividuo()
        mejorFitness = self.__calcularFitness(mejorIndividuo)

        return mejorIndividuo, mejorFitness
