import numpy as np
import matplotlib.pyplot as plt
import perceptronNN as pnn
import pandas as pd

# Initialise variables
population_size = 100
num_genes = 66
num_generations = 750
# Rate of crossover
cross_rate = 0.6
# Rate of mutation multiplied by 1000
mut_rate = 5
# Smell accuracy
smell_acc = 1

# Life length of creature, 10 eating iterations per unit
life_length = 4
# Network learning rate
Lrate = 0.2
runs = 1

colour_choice = ['r', 'b', 'g', 'y']


def fitness(genotype, life_length, environment, smell_acc, Lrate):
    """
        A mtethod to calculate a populations fitness by instantiating Perceptron NN from genotype
        :param genotype: A list containing an individuals bit string encoding their genes
        :param life_length: An integer
        :param environment: An integer
        :param smell_acc: A float
        :param Lrate: An integer
        :return fitness: A positive or negative integer, or 0
    """
    eat_cycle = 10 * life_length
    energy = []
    # Instantiate an array full of 1's and 0's to represent food and poison
    substance = np.zeros(eat_cycle)
    for index, _ in enumerate(substance):
        if index % 2 == 0:
            substance[index] = 0
        elif index % 2 != 0:
            substance[index] = 1
    substance = list(substance)

    # Instantiate neural network for given genotype
    p_network = pnn.PerceptronNN(genotype, smell_acc, Lrate)

    # Iterate through eating cycles
    for _ in range(eat_cycle):
        # Randomly select from substance array
        f_or_p = np.random.choice(substance)
        # Remove substance
        substance.remove(f_or_p)
        # Differentiate between fields
        if environment == 1:
            # Food is red in environment = 1
            if  f_or_p == 0:
                if np.random.rand() <= smell_acc:
                    sw_or_so = 0
                else:
                    sw_or_so = 1
                r_or_g = 0
            # Poison is green in environment = 1
            elif f_or_p == 1:
                if np.random.rand() <= smell_acc:
                    sw_or_so = 1
                else:
                    sw_or_so = 0
                r_or_g = 1 
        elif environment == 2:
            # Food is green in environment = 2
            if f_or_p == 0:
                if np.random.rand() <= smell_acc:
                    sw_or_so = 0
                else:
                    sw_or_so = 1
                r_or_g = 1
            # Poison is red in environment = 2
            elif f_or_p == 1:
                if np.random.rand() <= smell_acc:
                    sw_or_so = 1
                else:
                    sw_or_so = 0
                r_or_g = 0
        # Get NN output
        # If NN oupts 1, it eats
        # If NN outputs 0, it does not eat
        x = p_network.feedForward(sw_or_so, r_or_g)
        # # If NN eats and substance is food
        if x > 0 and f_or_p == 0:
            # Add energy
            energy.append(1)
        # If NN eats and substance is poison
        elif x > 0 and f_or_p == 1:
            # Add negative energy
            energy.append(-1)
        # If NN does not eat, add no energy
        elif x <= 0:
            energy.append(0)
        # energy.append(x)
        p_network.updateWeights()
    # At end of life sum energy stored, this is fitness
    # print(energy)
    fitness = sum(energy)

    return fitness


def rankPopulationByFitness(population):
    """
        A method to rank population by fitness
        :param population: A list of tuples containing fitness and corresponding genotype
        :return ranked_pop: A list of tuples containing genotype ranking and corresponding genotype
    """
    # Sort population by fitness
    population.sort(key= lambda tup:tup[0])
    
    ranked_pop = []
    # Enumerate through population and store index as ranking and genotype
    # No longer need the genotypes' fitness
    for index, tup in enumerate(population):
        ranked_pop.append((index, tup[1]))
    
    return ranked_pop


def crossover(ranked_pop, cross_rate):
    """
        A method to select two parents from the population with a probability proportional to their ranking.
        Selection done with roulette wheel
        :param ranked_pop: A list of tuples containing genotype ranking and corresponding genotype
        :param cross_rate: A float representing the rate of crossover
        :return offspring: A list containing offspring 
    """
    offspring = []
    # Create wheel to select 
    wheel = np.cumsum(range(population_size))
    max_wheel = sum(range(population_size))
    
    # Complete crossover for whole population
    for _ in range(population_size//2):

        # pick first individual 
        pick_1 = np.random.rand() * max_wheel 
        ind_1 = 1
        while pick_1 > wheel[ind_1]:
            ind_1 += 1
        
        
        # pick second individual 
        pick_2 = np.random.rand() * max_wheel 
        ind_2 = 1
        while pick_2 > wheel[ind_2]:
            ind_2 += 1
        
        # get fitness 
        parent_1 = ranked_pop[ind_1][1]
        
        parent_2 = ranked_pop[ind_2][1]
        if np.random.rand() <= cross_rate:
            # Create offspring from crossover 
            cross_over_point = np.random.choice(range(num_genes))

            parent_1_genes = parent_1[0:cross_over_point]
            parent_2_genes = parent_2[cross_over_point:]
            
            offspring_1 = np.hstack([parent_1_genes, parent_2_genes])
            offspring_2 = np.hstack([parent_2_genes, parent_1_genes])

            offspring.append(offspring_1)
            offspring.append(offspring_2)
        else:
            offspring_1 = parent_1
            offspring_2 = parent_2
            offspring.append(offspring_1)
            offspring.append(offspring_2)
    
    return offspring


def mutate(offspring, mut_rate):
    """
        A method to iterate through each offspring's genotype and potentially mutate
        :param offspring: A list containing lists of bit strings encoding a genotype
        :param mut_rate: A float representing mutation rate
        :return new_pop: A list containing a new population
    """
    new_pop = []
    # Iterate through population
    for genotype in offspring:
        # Enumerate through genotype
        for index, gene in enumerate(genotype):
            # If mutate, flip bit
            if np.random.randint(0,1000) <= mut_rate:
                if gene == 0:
                    genotype[index] = 1
                elif gene == 1:
                    genotype[index] = 0
        new_pop.append(genotype)
    return new_pop


def main():
    ave_fitness_over_runs = []
    for run in range(runs):
        ave_fitness_over_generations = [] 
        # Instantiate random population
        population = np.random.choice([0,1], size= (population_size, num_genes))
        # print(population)
        colour = colour_choice[run]
        # Iterate for the given number of generations
        for gen in range(num_generations):
            
            # Calculate fitness for population
            pop_fitness = [(fitness(genotype, life_length, np.random.choice([1,2]), smell_acc, Lrate), genotype) for genotype in population]

            # Calculate and store average fitness
            only_fitness = [i[0] for i in pop_fitness]

            sum_fitness = sum(only_fitness)
            
            ave_fitness = sum_fitness / population_size

            ave_fitness_over_generations.append(ave_fitness)
            ave_fitness_over_runs.append((ave_fitness, gen))


            # Rank population by fitness
            ranked_pop = rankPopulationByFitness(pop_fitness)

            # Potentially perform crossover, get offspring
            offspring = crossover(ranked_pop, cross_rate)
            
            # Potentially mutate offspring, get new populations
            new_pop = mutate(offspring, cross_rate)

            population = new_pop
           
        plt.plot(range(num_generations), ave_fitness_over_generations, color= colour, linewidth=0.5)
        
    
    # ave_fitness_over_runs.sort(key=lambda tup:tup[1])

    # sorted_fit = [i[0] for i in ave_fitness_over_runs]
    # sorted_gen = [i[1] for i in ave_fitness_over_runs]
    # rolling_fitness = pd.DataFrame(sorted_fit).rolling(500).mean()

    #plt.plot(sorted_gen, rolling_fitness, color='k')
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title(f"{num_generations} Generations, Runs: {runs}, Life length: {life_length}, Smell Accuracy: {smell_acc:.2f}")
    plt.show()

  


if __name__ == "__main__":
    main()
    