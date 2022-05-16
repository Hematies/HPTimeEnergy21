import scipy.io as scipy
from numpy import number
import math
from modelos.ModeloGeneral import  ModeloGeneral
import numpy as np

import pandas as pd
from copy import copy

# Esta clase implementa el modelo de ajuste V1:
class Modelo(ModeloGeneral):

    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="V1"):
        super().__init__(nombreFicheroMat)
        self.nombreModelo = nombreModelo
        self.matrizCluster = matrizCluster
        self.tablaTraduccionVariables = tablaTraduccionVariables

        # Añadimos algunas variables necesarias al workspace:
        self.datos['tasaCruce'] = 0.75
        self.datosOriginales['tasaCruce'] = 0.75

        self.datosSimulacion = None

    # Función privada para calcular la distribución de carga (Round Robin) del clúster:
    def __calcularDistribucionCarga(self, numSubpoblacionesTotales):
        res = []

        # En esta versión se aplica una distribución de carga simple:
        # Se distribuyen las subpoblaciones de manera uniforme. Si hay una etapa con más dispositivos que subpoblaciones,
        # se sigue prioridad de orden establecido en matrizCluster:

        # Etapa = Momento en el que hay disponible una o más subpoblaciones sin computar para una o más dispositivos disponibles.
        numDispositivosTotales = 0
        for indiceNodo in range(0, len(self.matrizCluster)):
            numDispositivosTotales = numDispositivosTotales + self.matrizCluster[indiceNodo]
            res.append([])

        # Calculamos el número de etapas "uniformes":
        numEtapasUniformes = int(numSubpoblacionesTotales / numDispositivosTotales)
        numSubpoblacionesExtras = numSubpoblacionesTotales % numDispositivosTotales

        # Rellenamos las subonblaciones en la matriz de carga:
        for indiceNodo in range(0, len(self.matrizCluster)):
            for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
                extra = 1 if numSubpoblacionesExtras > 0 else 0
                numSubpoblacionesExtras = numSubpoblacionesExtras - extra
                numSubpoblaciones = numEtapasUniformes + extra
                res[indiceNodo].append(numSubpoblaciones)

        return res

    # Función para calcular el coste temporal del dispositivo indicado:
    def __calcularCosteTiempoDispositivo(self, numSubpoblacionesTotales, numSubpoblaciones, indiceNodo, indiceDispostivo):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        # numIndSubp = (D['N'] / numSubpoblacionesTotales) * (3 - D['tasaCruce'])
        numIndSubp = (D['N'] / numSubpoblacionesTotales) * (1 + D['tasaCruce'])

        # Vamos a asumir por ahora que no hay migraciones locales.

        if D['P_'+i+'_'+j] > 0:
            tiempo = D['gen'] * numSubpoblaciones * (math.ceil(numIndSubp/D['P_'+i+'_'+j]) * D['W_'+i+'_'+j] / D['F_'+i+'_'+j])
        else:
            tiempo = 0

        return tiempo

    # Función para calcular el coste energético del dispositivo indicado:
    def __calcularCosteEnergiaDispositivo(self, tiempoNodo, tiempoDisp, indiceNodo, indiceDispostivo):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        # Vamos a asumir por ahora que no hay migraciones locales:

        tiempoIdle = tiempoNodo - tiempoDisp
        energia = D['Pow_'+i+'_'+j] * tiempoDisp + D['Pow_'+i+'_'+j+'_idle'] * tiempoIdle
        return energia

    # Función para calcular los costes de un nodo indicado:
    def __calcularCosteTiempoEnergiaNodo(self, numSubpoblacionesTotales,indiceNodo, matrizCarga):
        res = dict()
        D = self.datos  # Para abreviar el código
        i = str(indiceNodo)
        tiemposDispositivos = []

        # Calculamos el tiempo de ejecución de los dispositivos del nodo:
        tiempoNodo = 0.0
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            numSubpoblaciones = matrizCarga[indiceNodo][indiceDisp]
            costeTemporalDisp = \
                self.__calcularCosteTiempoDispositivo(numSubpoblacionesTotales, numSubpoblaciones, indiceNodo+1, indiceDisp+1)
            tiemposDispositivos.append(costeTemporalDisp)
            if costeTemporalDisp >= tiempoNodo:
                tiempoNodo = costeTemporalDisp

        # Calculamos la energía consumida de los dispositivos del nodo:
        energiaDispositivosTotal = 0.0
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            energiaDispositivosTotal = energiaDispositivosTotal \
                + self.__calcularCosteEnergiaDispositivo(tiempoNodo, tiemposDispositivos[indiceDisp],
                                                         indiceNodo+1, indiceDisp+1)

        # Finalmente, calculamos la energía como la energía de los dispositivos en ejecución más
        # el consumo en estado ocioso durante el tiempo de overhead y comunicación:
        potenciaIdleTotal = 0.0
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            potenciaIdleTotal = potenciaIdleTotal + D['Pow_'+str(indiceNodo+1)+'_'+str(indiceDisp+1)+'_idle']
        energiaNodo = energiaDispositivosTotal + (D['Tmaster'] + D['Tcom']) * potenciaIdleTotal
        # energiaNodo = energiaDispositivosTotal # Cambio de 10/08/21

        return tiempoNodo, energiaNodo

    # Función para renombrar los parámetros del modelo a una nomenclatura diferente:
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

        # Traducimos los nombres de las variables para que sean acordes a los modelos míos:
        self.__traducirVariablesWorkspace()

    # Método para computar el coste temporal y energético total predichos por el modelo:
    def computarCosteTiempoEnergia(self, numSubpoblacionesTotales, usarDatosModelo=False):
        res = dict()

        # if not self.tablaTraduccionVariables:
        if not usarDatosModelo:
            # Traducimos las variables del workspace de MATLAB a las variables aquí modeladas:
            self.__traducirVariablesWorkspace()
        D = self.datos  # Para abreviar el código

        # Calculamos la distribución de carga:
        matrizCarga = self.__calcularDistribucionCarga(numSubpoblacionesTotales)
        costeTiempoEnergiaNodos = []

        # Recopilamos los costes de los nodos:
        tiempoMaxNodos = 0.0
        energiaTotalNodos = 0.0
        for indiceNodo in range(0, len(self.matrizCluster)):
            costeTiempoEnergiaNodos.append(
                self.__calcularCosteTiempoEnergiaNodo(numSubpoblacionesTotales, indiceNodo, matrizCarga))

            # Apuntamos el tiempo máximo de cómputo desde nodo:
            if(costeTiempoEnergiaNodos[indiceNodo][0] >= tiempoMaxNodos):
                tiempoMaxNodos = costeTiempoEnergiaNodos[indiceNodo][0]

            # Apuntamos la energia total de los nodos:
            energiaTotalNodos = energiaTotalNodos + costeTiempoEnergiaNodos[indiceNodo][1]

        # Calculamos el coste temporal del cluster:
        res['T'] = D['NGmig']*(D['Tmaster'] + tiempoMaxNodos + D['Tcom'])

        # Calculamos el coste energético del cluster:
        res['energy'] = D['NGmig'] * (D['POW_cpu0'] * D['Tmaster'] + energiaTotalNodos + D['POW_sw'] * D['Tcom'])

        return res

    # Función para comprar si los parámetros dados como genes son consistentes:
    def esSolucionConsistente(self,genes):
        res = True

        for i in range(0, len(self.matrizCluster)):
            # Las potencias idle no pueden ser mayores que las activas:
            inconsistenciaGPU = genes[i*6 + 2] < genes[i*6 + 4]
            inconsistenciaCPU = genes[i*6 + 3] < genes[i*6 + 5]

            if inconsistenciaCPU or  inconsistenciaGPU:
                res = False
        return res

    ## Funciones de extracción de extracción de coste de variables (independiente):
    # PD: Una variable NO es un parámetro: los parámetros son las constantes a ajustar en el modelo
    def calcularCostesVariable(self, nombreVariable, cotaInferior, cotaSuperior, numMuestras=100):

        # Aseguramos que las cotas están bien establecidas:
        if cotaInferior < 0:
            cotaInferior = 0
        elif cotaInferior > cotaSuperior:
            cotaInferior = cotaSuperior

        resolucion = (cotaSuperior - cotaInferior) / numMuestras
        # listaCostesReales = []
        # listaCostesComputados = []
        listasDiferenciasTiempo = []
        listasDiferenciasEnergia = []

        for i in range(0, numMuestras):
            costesComputadosOriginalmente = np.array(self.calcularFitness([], devolverResultados=True, usarDatosModelo=True)[1])
            self.datos[nombreVariable] = cotaInferior + (i * resolucion)
            #costesReales, costesComputados = self.calcularFitness([], devolverResultados=True, usarDatosModelo=True)
            costesComputados = np.array(self.calcularFitness([], devolverResultados=True, usarDatosModelo=True)[1])
            # listaCostesReales.append(costesReales)

            listaDiferenciasTiempo = \
                list(map(lambda i: costesComputados[i]['T'] - costesComputadosOriginalmente[i]['T'],
                         range(len(costesComputados))))
            listasDiferenciasTiempo.append(listaDiferenciasTiempo)
            listaDiferenciasEnergia = \
                list(map(lambda i: costesComputados[i]['energy'] - costesComputadosOriginalmente[i]['energy'],
                         range(len(costesComputados))))
            listasDiferenciasEnergia.append(listaDiferenciasEnergia)

            # Corregimos el cambio:
            self.datos[nombreVariable] = self.datosOriginales[nombreVariable]

        #return listaCostesReales, listaCostesComputados
        return listasDiferenciasTiempo, listasDiferenciasEnergia

    # Función para predecir los costes al indicar un núm. de núcleos activos para un dispositivo concreto:
    def calcularCostesNumNucDisp(self, indiceNodo, indiceDisp, cotaInferior, cotaSuperior, numMuestras=100, parametros=None):
        if not parametros == None:
            self.introducirParametros(parametros, esGen=True)

        return self.calcularCostesVariable('P_' + str(indiceNodo+1) + '_' + str(indiceDisp+1), cotaInferior, cotaSuperior, numMuestras)
            # Nota: Si cotaInferior = 0, las primeras simulaciones son para el apagado del dispositivo correspondiente.

    # Función para predecir los costes al indicar un núm. de individuos por subpoblación:
    def calcularCostesNumIndPorSubp(self, cotaInferior, cotaSuperior, numMuestras=100, parametros=None):
        if not parametros == None:
            self.introducirParametros(parametros, esGen=True)
        return self.calcularCostesVariable('N', cotaInferior, cotaSuperior, numMuestras)

    # Función para predecir los costes al dejar inactivo un nodo en concreto:
    def calcularCostesApagarNodo(self, indiceNodo, parametros=None):
        datosOrig = dict()

        if not parametros == None:
            self.introducirParametros(parametros, esGen=True)

        costesComputadosOriginalmente = self.calcularFitness([], devolverResultados=True, usarDatosModelo=True)[1]
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            j = str(indiceDisp)

            datosOrig['P_' + str(indiceNodo+1) + '_' + str(indiceDisp+1)] = self.datos['P_' + str(indiceNodo+1) + '_' + str(indiceDisp+1)]
            self.datos['P_' + str(indiceNodo+1) + '_' + str(indiceDisp+1)] = 0

        costesComputados = np.array(self.calcularFitness([], devolverResultados=True, usarDatosModelo=True)[1])

        listaDiferenciasTiempo = \
            list(map(lambda i : float(costesComputados[i]['T'] - costesComputadosOriginalmente[i]['T']),range(len(costesComputados))))
        listaDiferenciasEnergia = \
            list(map(lambda i: float(costesComputados[i]['energy'] - costesComputadosOriginalmente[i]['energy']),
                     range(len(costesComputados))))

        # Corregimos el cambio:
        for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
            j = str(indiceDisp)
            self.datos['P_' + str(indiceNodo+1) + '_' + str(indiceDisp+1)] = datosOrig['P_' + str(indiceNodo+1) + '_' + str(indiceDisp+1)]
        return listaDiferenciasTiempo, listaDiferenciasEnergia

    # Método/Función para simular el clúster variando el núm. de núcleos activos y así construir el espacio de configuraciones:
    def simulacionCostesNumNucDisp(self, parametros=None, resolucion=4):
        # Es necesario establecer una resolución, porque de lo contrario el número de muestras es elevadísimo (No Polinómico)

        if not parametros == None:
            self.introducirParametros(parametros, esGen=True)

        # Guardamos los nombres de las columnas:
        columnas = []
        for indiceNodo in range(0, len(self.matrizCluster)):
            for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
                columnas.append('P_' + str(indiceNodo + 1) + '_' + str(indiceDisp + 1))
        columnas.extend(['numSubpoblaciones', 'costeTemporal', 'costeEnergetico'])

        # Recolectamos todas las combinaciones posibles:
        combinacionesPosibles = self.__combinacionesNumNucDisp(resolucion=resolucion)

        # Calculamos los costes temporales y energéticos de las diferentes combinaciones.
        # Tupla: (Configuración núcleos de disp, Num subpoblaciones, Coste temporal, Coste energético)
        resultados = []
        datosPreviosSimulacion = copy(self.datos)
        numMaxSubpoblaciones = self.numMaxSubpoblaciones
        for combinacion in combinacionesPosibles:
            # Preparamos los parámetros para ser computada la simulación:
            parametros = self.datos
            indiceNumNucleos = 0
            for indiceNodo in range(0, len(self.matrizCluster)):
                for indiceDisp in range(0, self.matrizCluster[indiceNodo]):
                    parametros['P_' + str(indiceNodo + 1) + '_' + str(indiceDisp + 1)] = combinacion[indiceNumNucleos]
                    indiceNumNucleos = indiceNumNucleos + 1
            self.datos = parametros

            # Calculamos los costes para todas las subpoblaciones posibles:
            for numSubpoblaciones in range(1, numMaxSubpoblaciones):
                costes = self.computarCosteTiempoEnergia(numSubpoblaciones, usarDatosModelo=True)
                print(combinacion, " ", numSubpoblaciones, " ", costes)

                # Almacenamos los resultados:
                resultados.append([])
                resultados[-1].extend(combinacion)
                resultados[-1].extend([numSubpoblaciones, float(costes['T']), float(costes['energy'])])

        self.datos = copy(datosPreviosSimulacion)

        # Escribimos los resultados en un dataframe interno y lo devolvemos:
        self.datosSimulacion = pd.DataFrame(data=resultados, columns=columnas)

        return self.datosSimulacion

    # Método para obtener todas las combinaciones posibles de núm. de núcleos activos de dispositivos dada una resolución:
    def __combinacionesNumNucDisp(self, resolucion):
        # Combinación imposible: (0,0,...,0)

        # Dimensiones de la matriz de combinaciones: MxN
        # M = Número de combinaciones posibles = Multiplicación de todos los números de nucleos (incluyendo casos de 0 núcleos):
        M = 1
        # Lista de tuplas de índices nodo-dispositivo como cadenas:
        listaNombresVariables = []
        # N = Número total de dispositivos:
        N = 0

        multiplicatorio = [1]

        # Para casos de Núm. nucleos % resolucion != 0, definimos por cada dispositivo los números de núcleos a usar
        # para las combinaciones:
        numerosNucleosDisp = []

        for indiceNodo in range(0, len(self.matrizCluster)):
            for indiceDisp in range(0, self.matrizCluster[indiceNodo]):

                N = N + 1
                listaNombresVariables.append('P_' + str(indiceNodo + 1) + '_' + str(indiceDisp + 1))

                # Definimos los números de núcleos usados de este dispositivo para formar combinaciones:
                numeroNucleos = 0
                numerosNucleosDisp.append([])
                numeroTotalNucleos = int(self.datos['P_' + str(indiceNodo + 1) + '_' + str(indiceDisp + 1)])
                while numeroNucleos <= numeroTotalNucleos:
                    numerosNucleosDisp[-1].append(numeroNucleos)
                    numeroNucleos = numeroNucleos + resolucion
                # Si nos hemos saltado el último número, lo incluimos (así representamos rendimiento al 100%):
                if numerosNucleosDisp[-1][-1] != numeroTotalNucleos:
                    numerosNucleosDisp[-1].append(numeroTotalNucleos)

                M = M * len(numerosNucleosDisp[-1])

                multiplicatorio.append(M)

        # Rellenamos toda la matriz de combinaciones posibles:
        combinacionesPosibles = np.zeros((M, N))
        for j in range(0,N):
            # Para rellenar la tabla debe saberse cuántas veces repetimos cada valor en la columna antes de pasar al
            # siguiente:
            numSaltosEntreValores = multiplicatorio[j]
            m = 0 # Índice de la lista de números de núcleos
            numNucleosActivos = 0 # Número de núcleos activos finalmente

            for i in range(0, M):
                if ((i % numSaltosEntreValores) == 0) and i != 0:
                    m = m + 1
                    m = m % len(numerosNucleosDisp[j])
                    numNucleosActivos = numerosNucleosDisp[j][m]
                combinacionesPosibles[i][j] = numNucleosActivos


        # Eliminamos la única combinación que no es posible: (0,...,0)
        combinacionesPosibles = np.delete(combinacionesPosibles, 0, 0)
        # print(combinacionesPosibles)

        return combinacionesPosibles

    # Método para exportar los datos del espacio de configuraciones:
    def exportarDatosSimulacion(self):
        if self.datosSimulacion is not None:
            from datetime import datetime
            ahora = datetime.now()
            cadena = ahora.strftime("%d_%m_%Y_%H_%M")
            self.datosSimulacion.to_csv('datosSimulacion/'+self.nombreModelo+'_'+cadena+'.csv')
        else:
            raise Exception('¡No hay datos de simulación computados!')

    # Método para importar los datos del espacio de configuraciones:
    def importarDatosSimulacion(self, path):
        if path is not None:
            self.datosSimulacion = pd.read_csv(path)
        else:
            raise Exception('¡No se ha indicado path!')