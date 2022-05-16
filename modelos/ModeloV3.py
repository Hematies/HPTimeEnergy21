import scipy.io as scipy
from numpy import number
import math
from modelos.ModeloV2 import  Modelo as ModeloMioV2

# Clase que implementa el modelo matemático V3:
class Modelo(ModeloMioV2):

    def __init__(self, nombreFicheroMat, matrizCluster, tablaTraduccionVariables=dict(), nombreModelo="ModeloV3"):
        super().__init__(nombreFicheroMat, matrizCluster, tablaTraduccionVariables=tablaTraduccionVariables)
        self.nombreModelo = nombreModelo

    # Método para calcular el coste en energía del dispositivo dado:
    def __calcularCosteEnergiaDispositivo(self, tiempoDisp, indiceNodo, indiceDispostivo):
        D = self.datos
        i = str(indiceNodo)
        j = str(indiceDispostivo)

        tiempoIdle = self.tiempoTotal - tiempoDisp

        # Pow_i_j (NOMBRE A CORREGIR) -> Energía media de conmutación en un núcleo del disp ocupado j en el nodo i
        # Pow_i_j_idle (NOMBRE A CORREGIR) -> Energía media de conmutación en un núcleo del disp ocioso j en el nodo i

        # OJO: La energía se modela como alpha*c*(v^2)*f*t, donde la frecuencia es dependiente del voltaje.
        # Por lo tanto, sin saber cómo varía a priori la frecuencia con el voltaje, no podemos optimizar la frecuencia.
        energia = D['P_' + i + '_' + j] * D['F_' + i + '_' + j] \
                  * (D['Pow_' + i + '_' + j] * tiempoDisp + D['Pow_' + i + '_' + j + '_idle'] * tiempoIdle)
        return energia

    # Método para devolver los valores de los parámetros originalmente guardados en el workspace dado:
    def construirIndividuoOriginal(self):
        datos = self.datosOriginales
        parametros = []

        for indice in range(1, 4):
            i = str(indice)
            parametros.append(datos['Wgpu' + i] / self.MAX_VALOR_W)
            parametros.append(datos['Wcpu' + i] / self.MAX_VALOR_W)
            parametros.append(datos['POW_gpu' + i] / self.MAX_VALOR_E)
            parametros.append(datos['POW_cpu' + i] / self.MAX_VALOR_E)
            parametros.append(datos['POW_gpu' + i + '_idle'] / self.MAX_VALOR_E)
            parametros.append(datos['POW_cpu' + i + '_idle'] / self.MAX_VALOR_E)

        # Otros parámetros a ajustar:
        parametros.append(datos['Tcom'] / self.MAX_VALOR_T)
        parametros.append(datos['Tmaster'] / self.MAX_VALOR_T)
        parametros.append(datos['POW_cpu0'] / self.MAX_VALOR_P)
        parametros.append(datos['POW_sw'] / self.MAX_VALOR_P)
        return parametros

    # Método para mostrar los valores de genes (individuo) que maneja el modelo:
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
                    nombreVariable = "Pow_"+dispositivo+str(int(i / 6) + 1)
                    if(i % 6 >= 4):
                        nombreVariable = nombreVariable + "_idle"
                    print(nombreVariable+"=", individuo[i] * self.MAX_VALOR_E, ";% vs ",
                          (datosOriginales[i] * self.MAX_VALOR_E))
            else:
                print("Tcom=",individuo[i] * self.MAX_VALOR_T, ";% vs ", (datosOriginales[i] * self.MAX_VALOR_T))
                print("Tmaster=", individuo[i+1] * self.MAX_VALOR_T, ";% vs ", (datosOriginales[i+1] * self.MAX_VALOR_T))
                print("POW_cpu0=", individuo[i+2] * self.MAX_VALOR_P, ";% vs ", (datosOriginales[i+2] * self.MAX_VALOR_P))
                print("POW_sw=", individuo[i+3] * self.MAX_VALOR_P, ";% vs ", (datosOriginales[i+3] * self.MAX_VALOR_P))
                break
            i = i + 1

    # Método para calcular los costes del nodo dado:
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
            potenciaIdleTotal = potenciaIdleTotal
            + D['P_'+str(indiceNodo+1)+'_'+str(indiceDisp+1)] * D['F_'+str(indiceNodo+1)+'_'+str(indiceDisp+1)] \
            * D['Pow_'+str(indiceNodo+1)+'_'+str(indiceDisp+1)+'_idle']
        energiaNodo = energiaDispositivosTotal + (D['Tmaster']+D['Tcom']) * potenciaIdleTotal

        return energiaNodo

    # Método para calcular el fitness para los valores de genes dados:
    def calcularFitness(self, genes, devolverResultados=False):
        numMaxSubpoblaciones = self.numMaxSubpoblaciones
        modelo = self
        datos = self.datos

        # Convertimos los genes en un diccionario de parámetros:
        parametros = dict()
        j = 0
        for indice in range(1, 4):
            i = str(indice)
            parametros['Wgpu' + i] = int(genes[j] * self.MAX_VALOR_W)
            parametros['Wcpu' + i] = int(genes[j + 1] * self.MAX_VALOR_W)
            parametros['POW_gpu' + i] = genes[j + 2] * self.MAX_VALOR_E
            parametros['POW_cpu' + i] = genes[j + 3] * self.MAX_VALOR_E
            parametros['POW_gpu' + i + '_idle'] = genes[j + 4] * self.MAX_VALOR_E
            parametros['POW_cpu' + i + '_idle'] = genes[j + 5] * self.MAX_VALOR_E
            j = j + 6
        parametros['Tcom'] = genes[j] * self.MAX_VALOR_T
        parametros['Tmaster'] = genes[j + 1] * self.MAX_VALOR_T
        parametros['POW_cpu0'] = genes[j + 2] * self.MAX_VALOR_P
        parametros['POW_sw'] = genes[j + 3] * self.MAX_VALOR_P
        modelo.introducirParametros(parametros)

        # Recogemos los costes reales:
        tiemposReales = datos["time_v1v2v3_all"][:, 2]
        energiasReales = datos["energ_v1v2v3_all"][:, 2]

        errorCuadrMedioTiempo = 0.0
        errorCuadrMedioEnergia = 0.0

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

            errorCuadrMedioTiempo = errorCuadrMedioTiempo + \
                                    (1 / numMaxSubpoblaciones) * math.pow(costes['T'] - tiempoReal, 2)
            errorCuadrMedioEnergia = errorCuadrMedioEnergia + \
                                     (1 / numMaxSubpoblaciones) * math.pow(costes['energy'] - energiaReal, 2)

        if not devolverResultados:
            return errorCuadrMedioTiempo, errorCuadrMedioEnergia
        else:
            return costesReales, costesComputados 
