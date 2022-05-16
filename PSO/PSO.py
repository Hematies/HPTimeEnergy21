import pyswarms as ps
from pyswarms.backend.topology.ring import Ring
import numpy as np

# Función que define el modelo PSO:
class PSO:
    def __init__(self, modelo,opciones=None, numParticulas=100,numIteraciones=80, ):

        self.modelo = modelo
        numDimensiones = len(modelo.construirIndividuoOriginal())
        if opciones is None:
            # opciones = {'c1': 0.5, 'c2': 0.3, 'w': 0.7}
            opciones = {'c1': 0.5, 'c2': 0.3, 'w': 0.7, 'k': int(numParticulas / 5), 'p': 2}
        limites = (np.ones(numDimensiones) * 0.0, np.ones(numDimensiones) *1.0)

        #self.optimizador = ps.single.GlobalBestPSO(n_particles=numParticulas, dimensions=numDimensiones,
        #       options=opciones,bounds=limites)

        self.optimizador = ps.single.GeneralOptimizerPSO(n_particles=numParticulas, dimensions=numDimensiones,
                                                   options=opciones, bounds=limites, topology=Ring(),
                                                   # init_pos=np.zeros((numParticulas, numDimensiones)),
                                                   init_pos=np.array([np.asarray(modelo.construirIndividuo()).reshape((-1))]
                                                                     *numParticulas),
                                                   bh_strategy='reflective')
                                                   #init_pos=None)

        self.numIteraciones = numIteraciones

    # Función que ejecuta el algoritmo de PSO:
    def calcularPSO(self):
        coste, individuo = self.optimizador.optimize(self.modelo.calcularFitness, iters=self.numIteraciones, verbose=True,
                                                     PSO=True)
        return individuo