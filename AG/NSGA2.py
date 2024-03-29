#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

# Código adaptado a partir del ejemplo de:
# https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

# Función que define valores uniformes entre dos cotas:
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

# Clase que implementa un modelo NSGA2:
class NSGA2:

    def __init__(self, modelo,
                 NGEN=100, MU=120,
                 modeloOptimizacionReparto=False,
                 modeloOptimizacionNucleosActivos=False):# NGEN=150, MU=100):

        toolbox = base.Toolbox()
        self.NGEN = NGEN
        self.MU = MU

        # Problem definition
        # Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
        BOUND_LOW, BOUND_UP = 0.0, 1.0

        # Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
        # BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

        # Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10

        if not modeloOptimizacionReparto:
            if not modeloOptimizacionNucleosActivos:
                from modelos.ModeloGeneral import numParametrosIndividuo
            else:
                numParametrosIndividuo = modelo.numTotalDispositivos * 2
        elif not modeloOptimizacionNucleosActivos:
            numParametrosIndividuo = modelo.numTotalDispositivos * 2


        NDIM = numParametrosIndividuo

        similitudPadreHijo = 0.75

        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, modelo.construirIndividuo)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", modelo.calcularFitness)

        if not modeloOptimizacionReparto and not modeloOptimizacionNucleosActivos:
            toolbox.decorate("evaluate", tools.DeltaPenalty(modelo.esSolucionConsistente, 10e9))

        # Bounded crossover:
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=similitudPadreHijo)

        PROBABILIDAD_MUTACION = 0.005
        toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=similitudPadreHijo,
                         # indpb = PROBABILIDAD_MUTACION)
                         indpb=1.0 / NDIM)


        toolbox.register("select", tools.selNSGA2)

        self.toolbox = toolbox
        self.modelo = modelo

    # Función que ejecuta el algoritmo NSGA2:
    def calcularNSGA2(self, seed=None, NGEN=None, MU=None,):
        random.seed(seed)
        toolbox = self.toolbox

        if NGEN is None:
            NGEN = self.NGEN
        if MU is None:
            MU = self.MU
        #CXPB = 0.9
        CXPB = 0.7
        # PROBABILIDAD_MUTACION_INICIAL = 0.9
        # PROBABILIDAD_MUTACION = PROBABILIDAD_MUTACION_INICIAL

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = toolbox.population(n=MU)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)

        # Begin the generational process
        for gen in range(1, NGEN):
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)

                '''
                if random.random() <= PROBABILIDAD_MUTACION:
                    toolbox.mutate(ind1)
                    toolbox.mutate(ind2)
                '''
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

                del ind1.fitness.values, ind2.fitness.values

            # PROBABILIDAD_MUTACION = PROBABILIDAD_MUTACION / 1.025

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            # Esto es para imprimir por pantalla:
            print(logbook.stream)


        print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

        return pop, logbook