import scipy.io as scipy
from numpy import number
import math
import pickle
import numpy as np
from AG.NSGA2 import NSGA2

from copy import copy

from sklearn.preprocessing import minmax_scale, scale,MinMaxScaler, StandardScaler



numParametrosIndividuo = 3*6+4


# Esta clase implementa un conjunto de métodos base que será utilizado en el resto de clases Modelo
class ModeloGeneral:

    def __init__(self, nombreFicheroMat, parametros={}, numMaxSubp=32, nombreModelo="ModeloGeneral"):
        # Leemos el fichero .mat y lo cargamos en el objeto
        self.nombreFicheroMat = nombreFicheroMat
        self.datos = scipy.loadmat(nombreFicheroMat) # self.datos

        # Corregimos las potencias de las CPUs:
        self.datos['POW_cpu0'] = 1 * 0.092;
        self.datos['POW_cpu1'] = 1 * 0.092;
        self.datos['POW_cpu2'] = 2 * 0.092;
        self.datos['POW_cpu3'] = 4 * 0.092;

        # Por si el workspace no tiene inicializado el resto de potencias:
        self.datos['POW_gpu0'] = 0.092;
        self.datos['POW_gpu1'] = 0.092;
        self.datos['POW_gpu2'] = 0.092;
        self.datos['POW_gpu3'] = 0.092;
        self.datos['POW_gpu0_idle'] = 0.092;
        self.datos['POW_gpu1_idle'] = 0.092;
        self.datos['POW_gpu2_idle'] = 0.092;
        self.datos['POW_gpu3_idle'] = 0.092;
        self.datos['POW_cpu0_idle'] = 1 * 0.092;
        self.datos['POW_cpu1_idle'] = 1 * 0.092;
        self.datos['POW_cpu2_idle'] = 2 * 0.092;
        self.datos['POW_cpu3_idle'] = 4 * 0.092;
        self.datos['POW_sw'] = 0.092;
        self.datos['POW_sw_idle'] = 0.092;


        # Guardamos los parámetros originales:
        self.datosOriginales = copy(self.datos)
        # self.datosOriginales = self.datos

        # Introducimos los genes (parámetros a ajustar del modelo) si los hay:
        if(not (parametros == {})):
            self.introducirParametros(parametros)

        # Introducimos los parámetros para el fitness:
        self.numMaxSubpoblaciones = numMaxSubp

        # Definimos las cotas de las variables de número de ciclos, potenica, energía y tiempo:
        self.MAX_VALOR_W = math.pow(10, 9)# 7)
        self.MAX_VALOR_P = 10.0
        # self.MAX_VALOR_P = 0.01
        #self.MAX_VALOR_P = math.pow(10, -8)
        self.MAX_VALOR_E = math.pow(10, -8)  # Al parecer la escala cuenta bastante para el algoritmo genético
        # CUIDADO: EN EL MODELO MÍO V3 HAY POTENCIA PARA POW_SW, ETC POR UN LADO Y ENERGIA DE CONMUTACION POR OTRO
        self.MAX_VALOR_T = 20# 200.0

        # Nombre del modelo (de cara a importar y exportar):
        self.nombreModelo = nombreModelo

    # Método para introducir parámetros dado en el modelo:
    def introducirParametros(self, parametros, esGen=False):
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


    # Método para devolver los valores de los parámetros originalmente guardados en el workspace dado:
    def construirIndividuoOriginal(self):
        datos = copy(self.datosOriginales)
        parametros = []

        for indice in range(1, 4):
            i = str(indice)
            parametros.append(datos['Wgpu' + i] / self.MAX_VALOR_W)
            parametros.append(datos['Wcpu' + i] / self.MAX_VALOR_W)
            parametros.append(datos['POW_gpu' + i] / self.MAX_VALOR_P)
            parametros.append(datos['POW_cpu' + i] / self.MAX_VALOR_P)
            parametros.append(datos['POW_gpu' + i + '_idle'] / self.MAX_VALOR_P)
            parametros.append(datos['POW_cpu' + i + '_idle'] / self.MAX_VALOR_P)

        # Otros parámetros a ajustar:
        parametros.append(datos['Tcom'] / self.MAX_VALOR_T)
        parametros.append(datos['Tmaster'] / self.MAX_VALOR_T)
        parametros.append(datos['POW_cpu0'] / self.MAX_VALOR_P)
        parametros.append(datos['POW_sw'] / self.MAX_VALOR_P)
        return parametros

    # Método para devolver los valores de los parámetros (individuo) que maneja el modelo:
    def construirIndividuo(self):
        datos = self.datos
        parametros = []

        for indice in range(1, 4):
            i = str(indice)
            parametros.append(datos['Wgpu' + i] / self.MAX_VALOR_W)
            parametros.append(datos['Wcpu' + i] / self.MAX_VALOR_W)
            parametros.append(datos['POW_gpu' + i] / self.MAX_VALOR_P)
            parametros.append(datos['POW_cpu' + i] / self.MAX_VALOR_P)
            parametros.append(datos['POW_gpu' + i + '_idle'] / self.MAX_VALOR_P)
            parametros.append(datos['POW_cpu' + i + '_idle'] / self.MAX_VALOR_P)

        # Otros parámetros a ajustar:
        parametros.append(datos['Tcom'] / self.MAX_VALOR_T)
        parametros.append(datos['Tmaster'] / self.MAX_VALOR_T)
        parametros.append(datos['POW_cpu0'] / self.MAX_VALOR_P)
        parametros.append(datos['POW_sw'] / self.MAX_VALOR_P)
        return parametros

    # Método para mostrar los valores de los parámetros (individuo) que maneja el modelo:
    def mostrarValoresIndividuo(self, individuo):

        datosOriginales = self.construirIndividuoOriginal()
        i=0
        for gen in individuo:
            dispositivo = "gpu" if (i % 2 == 0) else "cpu"
            if i < 18:
                if i % 6 < 2:
                    nombreVariable = "W"+dispositivo+str(int(i / 6) + 1)
                    print(nombreVariable+"=", (int)(individuo[i] * self.MAX_VALOR_W), ";% vs ",
                          (int)(datosOriginales[i] * self.MAX_VALOR_W))
                else:
                    nombreVariable = "POW_"+dispositivo+str(int(i / 6) + 1)
                    if(i % 6 >= 4):
                        nombreVariable = nombreVariable + "_idle"
                    print(nombreVariable+"=", individuo[i] * self.MAX_VALOR_P, ";% vs ",
                          (datosOriginales[i] * self.MAX_VALOR_P))
            else:
                print("Tcom=",individuo[i] * self.MAX_VALOR_T, ";% vs ", (datosOriginales[i] * self.MAX_VALOR_T))
                print("Tmaster=", individuo[i+1] * self.MAX_VALOR_T, ";% vs ", (datosOriginales[i+1] * self.MAX_VALOR_T))
                print("POW_cpu0=", individuo[i+2] * self.MAX_VALOR_P, ";% vs ", (datosOriginales[i+2] * self.MAX_VALOR_P))
                print("POW_sw=", individuo[i+3] * self.MAX_VALOR_P, ";% vs ", (datosOriginales[i+3] * self.MAX_VALOR_P))
                break
            i = i + 1


    # Método para devolver los valores de fitness para un individuo (genes) dado, que es el conjunto de valores de ajuste:
    # devolverResultados=True: Devolver directamente los costes computados y reales
    # usarDatosModelo=True: No se imputan los genes en el modelo y se usan directamente los datos ya guardados
    def calcularFitness(self, genes, devolverResultados=False, usarDatosModelo=False):
        numMaxSubpoblaciones = self.numMaxSubpoblaciones
        modelo = self
        datos = self.datos

        if not usarDatosModelo:
            # Convertimos los genes en un diccionario de parámetros:
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


        if not devolverResultados:
            return errorCuadrMedioTiempo, errorCuadrMedioEnergia
        else:
            return costesReales, costesComputados


    ## Funciones de importación y exportación de parámetros del modelo:
    def exportarModelo(self, rutaCarpeta=None):
        from datetime import datetime
        ahora = datetime.now()
        cadena = ahora.strftime("%d_%m_%Y_%H_%M")
        if rutaCarpeta is None:
            rutaCarpeta = "ficherosModelos"
        pickle.dump(self, open(rutaCarpeta+"/"+self.nombreModelo+"_"+cadena+".mod", "wb"))

    @staticmethod # No tiene sentido reescribir el mismo modelo (por la jerarquía de clases definida)
    # Método para crear un modelo a partir de uno guardado en fichero:
    def importarModelo(rutaFichero):
        modeloImportado = pickle.load(open(rutaFichero, "rb"))
        return modeloImportado

    # Método para ajustar los parámetros del modelo:
    def ajustarParametros(self):
        algoritmoGenetico = NSGA2(self)
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

            v = listaFitness[i][0] + listaFitness[i][1]

            if ((mejorFitness == None) or (mejorFitness >= v)):
                mejorIndividuo = individuo
                mejorFitness = v
            i = i+1

        return mejorIndividuo, mejorFitness


