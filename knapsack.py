import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Session(object):
    def __init__(self, population_size, tournament_size, crossover_rate, mutation_rate, iterations):
        '''Constructor of Session object

            Attributes:
                population (Population): represents the current population
                task (Task): stores the task with the parameters of the knapsack problem
                number_of_items (int): number of items allowed in the knapsack
                population_size (int): maximum of individuals per population
                tournament_size (int): number of individuals compared in a tournament
                crossover_rate (float): [0, 1], probability of a crossover ocurring
                mutation_rate (float): [0,1], proportion of genes to mutate with respect of total number of genes (items)
                iterations (int): number of iterations of the simulation
                result (Individual): optimal solution found by the algorithm
                stats (Stats): object that stores and calculate the plots for visualization of the algorithm
        '''
        self.population = None
        self.task: Task
        self.number_of_items: int
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.result = None
        self. stats = None

    def generator(self, n, w, s, output_file):
        '''Creates a set of n random items and stores them in output_file

            Parameters:
                n (int): number of items in the problem
                w (float): maximum weight of the knapsack
                s (float): maximun load of the knapsack
                output_file (str): name of the file to dump the generated data
        '''
        assert n > 1000 and n < 2000, "Invalid input n (1000, 2000)"
        assert w > 10000 and n < 20000, "Invalid input w (10,000, 20,000)"
        assert s > 10000 and s < 20000, "Invalid input s (10,000, 20,000)"
        while True:
            w_sum = 0
            s_sum = 0

            with open(output_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow([n, w, s])
                for i in range(n):
                    rand_w = np.random.uniform(0.0, 10 * w/n)
                    rand_s = np.random.uniform(0.0, 10 * s/n)
                    rand_c = np.random.uniform(0.0, n)
                    w_sum += rand_w
                    s_sum += rand_s
                    writer.writerow([rand_w, rand_s, rand_c])
            if w_sum > 2*w and s_sum > 2*s:
                #Initial conditions are fulfilled, the document is valid
                break

    def read_task(self, input_file):
        '''Reads a csv file withe the knapsack problem parameters and creates a Task type object

            Parameters:
                input_file (str): name of the file that contains the data
        '''
        with open(input_file, 'r') as f:
            reader = csv.reader(f)
            header = list(map(int, next(reader)))
            self.number_of_items = header[0]
            #Create Task object with parameters n, w, s
            r = Task(header[0], header[1], header[2])
            #Load items into the task
            for row in reader:
                row = list(map(float, row))
                r.add_item(row[0], row[1], row[2])
        self.task = r

    def init_population(self):
        '''Creates Population type object of population_size random individuals'''
        pop = Population(self.population_size)
        for i in range(self.population_size):
            indv = Individual(self.number_of_items)
            pop.addindividual(indv)
        self.population = pop

    def tournament(self):
        '''Selects n = tournament_size random elements and finds the most fitted one.'''
        #Choose a random sample of individuals.
        chosen = np.random.choice(range(0, self.population.max_size), self.tournament_size, False) #Returns ndarray

        best = self.population.population[chosen[0]]
        best_fitness = best.fitness

        for i in range(1, self.tournament_size):
            ind = self.population.population[chosen[i]]
            if ind.fitness > best_fitness:
                #A better individual has been found
                best = ind
                best_fitness = ind.fitness
        return best

    def generate_new_population(self):
        '''Generates a Population object and fills it with population_size individuals.
            - Applies an elitism of e = 1, the best individual is automatically passed onto the next generation.
            - For the generation of a new individual, two random well fitted individuals are selected, crossover and mutation is performed'''
        new_population = Population(self.population_size)

        new_population.addindividual(self.best())
        while new_population.no_individuals < new_population.max_size:
            #Assert parents are different
            while True:
                parent1 = self.tournament()
                parent2 = self.tournament()
                if parent1.id != parent2.id:
                   break

            child = self.crossover(parent1, parent2)
            child2 = self.mutate(child)

            new_population.addindividual(child2)
        self.population = new_population

    def crossover(self, parent1, parent2):
        '''Performs crossover operation between the two individuals provided

            Parameters:
                parent1 (Individual): first individual
                parent2 (Individual): second individual
            Returns:
                Individual type object result of the crossover
        '''
        #Set sequence cutting point.
        cutting_point = 750

        if np.random.uniform(0.0, 1.0) < self.crossover_rate:
            #Crossover occurs
            child_seq = np.empty(self.number_of_items, dtype=bool)
            for i in range(self.number_of_items):
                if(i < cutting_point):
                    #Use parent1 sequence
                    child_seq[i] = parent1.representation[i]
                else:
                    #Use parent2 sequence
                    child_seq[i] = parent2.representation[i]
            child = Individual(self.number_of_items, child_seq)
            return child
        #Crossover does not occur.
        return parent1

    def mutate(self, individual):
        '''Performs mutation operation on the individual passed as parameter.
            Mutates n_items*mutation_rate genes

            Parameters:
                individual (Individual): individual to mutate
            Returns:
                Individual type object
                '''
        no_genes = int(self.number_of_items*self.mutation_rate)
        #Copy the individual to avoid aliasing
        res = Individual(self.number_of_items, individual.representation.copy())
        #Randomly select no_genes positions to mutate
        positions_to_mutate = np.random.choice(range(0, self.number_of_items), no_genes, False) #Returns ndarray

        for i in range(positions_to_mutate.size):
            #Position to mutate, perform NOT operation
            res.representation[positions_to_mutate[i]] = not res.representation[[positions_to_mutate[i]]]

        return res

    def best(self):
        '''Return the most fitted individual in the current population'''
        ret = []
        i = 0
        best = self.population.population[0]
        best_fitness = best.fitness

        while i < self.population_size:
            if self.population.population[i].fitness >= best_fitness:
                best = self.population.population[i]
                best_fitness = self.population.population[i].fitness
            i += 1
        return best

    def execute(self):
        '''Performs simulation under the conditions specified.

            Returns:
                Individual type object with the highest fitness of all the simulation.
        '''
        j = 0
        while j < self.iterations:
            print("Generation #" + str(j) + "...............")
            #Evaluate population
            self.population.evaluate(self.task)
            #Identify best inidividual
            b = self.best()
            #Add best individual of the generation for fitness plotting
            self.stats.add_best(b)
            print(b)

            #Select data to scatter
            bests = []
            for i in range(self.stats.resolution):
                bests.append(self.tournament())
            self.stats.add_scatter_fitness(bests)
            #Generate new population
            self.generate_new_population()
            j += 1
        #Evaluate last generation
        self.population.evaluate(self.task)
        self.res = self.best()
        self.stats.add_result(self.res)
        print("Results: sum_w = " + str(self.res.weight) + " sum_s = " + str(self.res.load) + " sum_w/W = " + str(self.res.weight/self.task.w) + " sum_s/S = " + str(self.res.load/self.task.s) + "\n")
        return self.res

    def show_stats(self):
        '''Plots the data gathered during simulation'''
        plt.show()

class Task(object):
    def __init__(self, n, w, s):
        '''Constructor of Task object

            Attributes:
                itemCounter (int): number of items loaded into the task
                n (int): number of items in the problem
                w (int): maximum weight of the knapsack
                s (int): maximum load of the knapsack
                items (ndarray(n,3)): array containing the properties of each item.
        '''
        self.itemCounter = 0
        self.n = n
        self.w = w
        self.s = s
        self.items = np.empty((n, 3))

    def add_item(self, w, s, c):
        '''Adds an item to the items list

            Parameters:
                w (float): individual weight of the item
                s (float): individual volume of the item
                c (float): individual value of the item
        '''
        self.items[self.itemCounter] = [w, s, c]
        self.itemCounter += 1

    def __str__(self):
        return "Number of items initialized: " + str(self.itemCounter) + "\nTotal number of items: " + str(self.n)

class Population:
    max_size: int
    no_individuals: int
    no_of_zeros: int

    def __init__(self, size):
        '''Constructor of Population object

            Attributes:
                no_individuals (int): number of individuals existing in the population.
                max_size (int): maximum number of individuals allowed in the population
                population (ndarray(max, size)): array of Individual type objects representing the whole population.
        '''
        self.no_individuals = 0
        self.max_size = size
        self.population = np.empty(self.max_size, dtype=object)

    def addindividual(self, individual):
        '''Adds a new individual to the population

            Parameters:
                individual (Individual): individual to be inserted into the population.
        '''
        self.population[self.no_individuals] = individual
        individual.id = self.no_individuals
        self.no_individuals += 1

    def evaluate(self, task):
        '''Performs fitness evaluation of the whole population

            Parameters:
                  task (Task): task object that will be used to evaluate the fitness of each individual.
        '''
        for i in range(self.no_individuals):
            self.population[i].evaluate(task)
        return 0

    def __str__(self):
        r = "Population size: " + str(self.max_size) + "---------------\n"
        for ind in range(self.no_individuals):
            r += self.population[ind].__str__()
        return r

class Individual(object):
    #Id in the population it belongs to
    id: None
    #Probability of an item existing in the knapsack represented by the Individual. Only for random initialization.
    p = 0.2
    def __init__(self, n_items: int, representation = np.empty(0)):
        '''Contructor of Individual object. Creates an individual with an item sequence randomly generated if no representation is provided\
            , or created from the items sequence passed as argument.
            Attributes:
                id (int): id of the individual that unqiely identifies it in the population
                representation (ndarray(n_items, dtqype = bool)): array of booleans representing the items in the knapsack.
                fitness (float): fitness of the individual
                value (float): total value of the knapsack, sum of all the values of its items.
                weight (float): total weight of the knapsack, sum of all the weights of its items.
                load (float): total load of the knapsack, sum of all the volumes of its items.

            Parameters:
                n_items (int): number of items in the knapsack
                representation (ndarray(n_items, dtqype = bool)): (Optional) array of booleans representing the knapsack.'''
        if representation.size == 0:
            #Generate a random individual
            self.representation = np.empty(n_items, dtype=bool)
            for i in range(n_items):
                self.representation[i] = True if np.random.uniform() < Individual.p else False
        else:
            assert representation.size == n_items, "Invalid representation, not item exhasutive"
            assert representation.dtype == bool, "Invalid type contained in ndarray"
            self.representation = representation
        self.fitness = None
        self.value = None
        self.weight = None
        self.load = None

    def evaluate(self, task: Task):
        '''Performs fitness evaluation of the individual. Assigns the value of the knapsack if the constraints
            are fulfilled, 0.0 otherwise.

            Parameters:
                task (Task): task object to be used in the evaluation.
            '''
        sum_w = 0.0
        sum_s = 0.0
        sum_c = 0.0
        #Obtain weight, value and load of the knapsack
        for i in range(task.n):
            if self.representation[i]:
                item = task.items[i]
                sum_w += item[0]
                sum_s += item[1]
                sum_c += item[2]

        self.value = sum_c
        self.weight = sum_w
        self.load = sum_s

        if sum_w <= task.w and sum_s <= task.s:
            #Constraints are fulfilled
            self.fitness = sum_c

        else:
            #Constraints are not fulfilled.
            self.fitness = 0.0

    def __str__(self):
        return "ID #" + str(self.id) + " Fitness: " + str(self.fitness) + "\n"\

class Stats(object):
    def __init__(self, task:Task, iterations, resolution):
        '''Constructor of Stats object. Stores data yielded for visualization and analysis purposes

            Parameters:
                task (Task): Task type object with the properties of the problem.
                iterations (int): number of iterations of the simulation.
                resolution (int): number of individuals that are going to be stored from each generation

            Attributes:
                iterations (int): number of iterations of the simulation
                resolution (int): number of individuals that are going to be stored from each generation
                task_data (list(3)): store of the problem parameters.
                bests (list): bests individuals of each generation.
                scatter_inds (list): selected individuals at each generation to be shown is scatter plot.
                result (Individual): result of the simulation.
        '''
        self.iterations = iterations
        self.resolution = resolution
        self.task_data = [task.n, task.s, task.w]

        self.bests = []
        self.scatter_inds = []
        self.result = None

    def add_best(self, best_ind):
        '''Inserts a individual in the bests list

        Parameters:
            best_ind (Individual): individual to be added
        '''
        self.bests.append([best_ind.fitness, best_ind])

    def add_scatter_fitness(self, bests_of_iteration):
        '''Inserts a individual in the scatter_inds list

            Parameters:
                best_ofs_iteration (list[float, Individual]): individual to be added
        '''
        self.scatter_inds.append(bests_of_iteration)

    def add_result(self, res):
        '''Inserts individual as result of simulation

            Parameters:
                res (Individual): individual to be added
        '''
        self.result = res

    def plot_fitness(self, label_, figure_id = None):
        '''Creates plot of the fitness over the iterations in a new figure, or in figure_id if provided

            Parameters:
                label (str): name of the chart
                figure_id (figure): figure in which to plot the chart.
        '''
        if figure_id != None:
            f = plt.figure(figure_id.number)
        else:
            f = plt.figure()
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Value")
        plt.plot(range(self.iterations), [self.bests[i][0] for i in range(self.iterations)], label = label_)
        plt.legend()
        return f

    def plot_scatter(self):
        '''Creates scatter plot of the gathered individuals at each iteration. Color pattern is used to indicate generation.
            Plots the relative individual weights with respect to the maximum weight on the x-axis
            Plots the relative individual loads with respect to the maximum load on the y-axis'''
        plt.figure()
        plt.xlabel("Wi/W")
        plt.ylabel("Si/S")
        colors = cm.get_cmap('rainbow')
        for j in range(self.iterations):
            for i in range(self.resolution):
                plt.scatter(self.scatter_inds[j][i].weight / self.task_data[2], self.scatter_inds[j][i].load / self.task_data[1],
                            color=colors(j / self.iterations), s=1.5)
        plt.scatter(self.result.weight / self.task_data[2], self.result.load / self.task_data[1], color='black')
