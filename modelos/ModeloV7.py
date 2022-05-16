import scipy.io as scipy
import numpy as np
import math
import random
from statistics import mean, stdev

from AG.NSGA2 import NSGA2
from modelos.ModeloV4 import  Modelo as ModeloMioV4
from modelos.ModeloV2 import  Modelo as ModeloMioV2
from modelos.ModeloV6 import  Modelo as ModeloMioV6

from copy import copy
from datetime import datetime

import pandas as pd
from sklearn.metrics import r2_score


MODELAR_TIEMPO_ENERGIA = 0
MODELAR_TIEMPO = 1
MODELAR_ENERGIA = 2

# Clase que implementa el modelo V7:
class Modelo(ModeloMioV6):


    # El modelo V7 es una extensión del V6 donde se puede aplicar CV.
    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloV5"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo
        self.MAX_VALOR_P = 10.0


        # Para poder hacer CV con este nuevo modelo, usamos las siguientes variables de control:
        self.__testFold = -1 # Cuando tengan un valor diferentes de -1, el cálculo de fitness se aplica teniendo en cuenta
            # que este fold es de test
        self.__trainFoldsEntrandose = False # Si vale True, el fitness se aplica a todos los folds que no sean self.testFold

        # Barajeo de folds:
        self.__barajaFolds = []

        self.numFolds = 5 # Un valor estándar a ser cambiado al llamar a aplicarCV()


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

        # Si no se aplica CV:
        if self.__testFold == -1:
            for numSubpoblaciones in range(1, numMaxSubpoblaciones + 1):
                # Calculamos los costes computados
                costes = modelo.computarCosteTiempoEnergia(numSubpoblaciones, usarDatosModelo=usarDatosModelo)
                costesComputados.append(costes)

                # Recogemos los costes reales para el núm de subp. dado:
                tiempoReal = tiemposReales[numSubpoblaciones - 1] / 1000  # Está en milisegudos
                energiaReal = energiasReales[numSubpoblaciones - 1]
                c = dict()
                c['T'] = tiempoReal
                c['energy'] = energiaReal
                costesReales.append(c)

                errorCuadrMedioTiempo = errorCuadrMedioTiempo + \
                                        (1 / numMaxSubpoblaciones) * math.pow(costes['T'] - tiempoReal, 2)
                errorCuadrMedioEnergia = errorCuadrMedioEnergia + \
                                         (1 / numMaxSubpoblaciones) * math.pow(costes['energy'] - energiaReal, 2)
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

        # Si se aplica CV:
        else:
            numSubpoblacionesPorFold = math.ceil(numMaxSubpoblaciones/self.numFolds)

            numSubpoblacionesEvaluadas = numMaxSubpoblaciones-numSubpoblacionesPorFold if self.__trainFoldsEntrandose else numSubpoblacionesPorFold

            for i in range(0, numMaxSubpoblaciones): # i: Índice en la baraja de folds
                # Guardamos el núm. de subpoblaciones de la baraja:
                numSubpoblaciones = self.__barajaFolds[i]

                # Computamos a qué fold pertenece:
                fold = int(i/ numSubpoblacionesPorFold)

                # Calculamos el fitness solo cuando estamos en el fold de test y estamos testeando,
                # o estamos en un fold que no es de test y estamos entrenando:
                if ((fold == self.__testFold) and (not self.__trainFoldsEntrandose)) \
                    or ((fold != self.__testFold) and (self.__trainFoldsEntrandose))\
                    or (self.numFolds == 1): # apañete rápido
                    # Calculamos los costes computados
                    costes = modelo.computarCosteTiempoEnergia(numSubpoblaciones, usarDatosModelo=usarDatosModelo)
                    costesComputados.append(costes)

                    # Recogemos los costes reales para el núm de subp. dado:
                    tiempoReal = tiemposReales[numSubpoblaciones - 1] / 1000  # Está en milisegudos
                    energiaReal = energiasReales[numSubpoblaciones - 1]
                    c = dict()
                    c['T'] = tiempoReal
                    c['energy'] = energiaReal
                    costesReales.append(c)

                    errorCuadrMedioTiempo = errorCuadrMedioTiempo + \
                                            (1 / numSubpoblacionesEvaluadas) * math.pow(costes['T'] - tiempoReal, 2)
                    errorCuadrMedioEnergia = errorCuadrMedioEnergia + \
                                             (1 / numSubpoblacionesEvaluadas) * math.pow(costes['energy'] - energiaReal, 2)


            # Apaños:
            errorCuadrMedioTiempo = math.sqrt(errorCuadrMedioTiempo)
            errorCuadrMedioEnergia = math.sqrt(errorCuadrMedioEnergia)

            # Un truco para devolver solo un error de los dos tipos:
            if (not self.tipoModelado == MODELAR_TIEMPO_ENERGIA) and self.__trainFoldsEntrandose:
                if self.tipoModelado == MODELAR_TIEMPO:
                    # return errorCuadrMedioTiempo, 1.0
                    errorCuadrMedioEnergia = 1.0
                elif self.tipoModelado == MODELAR_ENERGIA:
                    # return 1.0, errorCuadrMedioEnergia
                    errorCuadrMedioTiempo = 1.0


            # Si se está testeando, se devuelve tanto los errores como los costes:
            if not self.__trainFoldsEntrandose:
                # En caso de K-Fold-CV, normalizamos con los datos de test:
                if not self.numFolds == self.numMaxSubpoblaciones:
                    tiemposRealesFold = list(map(lambda coste: coste['T'], costesReales))
                    energiasRealesFold = list(map(lambda coste: coste['energy'], costesReales))
                    dispersionTiempo = stdev(
                        list(np.array(tiemposRealesFold) / 1000)) # Está en milisegudos
                    dispersionEnergia = stdev(energiasRealesFold)
                    errorCuadrMedioTiempo = errorCuadrMedioTiempo / dispersionTiempo
                    errorCuadrMedioEnergia = errorCuadrMedioEnergia / dispersionEnergia
                # En caso de LeaveOneOut-CV, probamos a normalizar el error con el conjunto total, no el de test:
                else:
                    dispersionTiempo = stdev(
                        list(np.array(tiemposReales) / 1000)) # Está en milisegudos
                    dispersionEnergia = stdev(energiasReales)
                    # print(errorCuadrMedioTiempo)
                    # print(errorCuadrMedioEnergia)
                    errorCuadrMedioTiempo = errorCuadrMedioTiempo / dispersionTiempo
                    errorCuadrMedioEnergia = errorCuadrMedioEnergia / dispersionEnergia

                return errorCuadrMedioTiempo, errorCuadrMedioEnergia, costesReales, costesComputados

        if not devolverResultados:
            return errorCuadrMedioTiempo, errorCuadrMedioEnergia
        else:
            return costesReales, costesComputados

    # Método para barajar los folds:
    def __barajarFolds(self):
        self.__barajaFolds = list(range(1,self.numMaxSubpoblaciones+1))
        random.shuffle(self.__barajaFolds)

    # Método que aplica validación cruzada LOO con un número de folds dado:
    def aplicarCV(self, numFolds, exportarModelos=False):

        if exportarModelos:
            contenidoDataframe = []

        # Apuntamos el núm. de folds usados:
        self.numFolds = numFolds

        # Barajamos los núms de subpoblaciones:
        self.__barajarFolds()

        # Entrenamos con CV este modelo por cada fold indicado:
        resultados = []
        datosAntesCV = copy(self.datos) # Guardamos los datos actuales por si acaso
        for i in range(0, numFolds):

            print("----- VAMOS CON EL FOLD ",i," -----")

            # Reiniciamos los datos:
            self.datos = copy(self.datosOriginales)

            # Actualizamos las variables de control del CV para entrenamiento:
            self.__trainFoldsEntrandose = True
            self.__testFold = i

            # Aplicamos entrenamiento:
            mejorIndividuo, mejorFitness = self.ajustarParametros()
            # self.introducirParametros(mejorIndividuo, esGen=True, datosPotenciaDisp=True)

            # Actualizamos las variables de control del CV para test:
            self.__trainFoldsEntrandose = False

            # Calculamos el fitness para dicho fold de test con los datos entrenados con el resto de folds:
            # errorCuadrMedioTiempo, errorCuadrMedioEnergia = self.calcularFitness(mejorIndividuo)
            errorCuadrMedioTiempo, errorCuadrMedioEnergia, costesReales, costesComputados = self.calcularFitness(mejorIndividuo)

            costesRealesTiempo = list(map(lambda coste: float(coste['T']), costesReales))
            costesRealesEnergia = list(map(lambda coste: float(coste['energy']), costesReales))
            costesComputadosTiempo = list(map(lambda coste: float(coste['T']), costesComputados))
            costesComputadosEnergia = list(map(lambda coste: float(coste['energy']), costesComputados))

            resultados.append((errorCuadrMedioTiempo, errorCuadrMedioEnergia))

            if exportarModelos:
                datos = [self.__barajaFolds[i]]
                datos.extend(mejorIndividuo)
                datos.append(errorCuadrMedioTiempo)
                datos.append(errorCuadrMedioEnergia)
                contenidoDataframe.append(datos)


        # Restauramos el modelo a su estado inicial:
        self.datos = copy(datosAntesCV)

        # Restauramos las variables de control del CV:
        self.__trainFoldsEntrandose = False
        self.__testFold = -1

        if exportarModelos:
            ahora = datetime.now()
            cadena = ahora.strftime("%d_%m_%Y_%H_%M")
            dataframe = pd.DataFrame(data=contenidoDataframe)  # , columns=columnas)
            dataframe.to_csv('LOO_CV_' + self.nombreModelo + '_' + cadena + '.csv')

        # Retornamos los resultados de test:
        return resultados








