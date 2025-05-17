import numpy as np
import pandas as pd
import random
import operator

# Create some cities (let's assume 20 random cities)
num_cities = 20
city_coordinates = np.random.rand(num_cities, 2) * 100  # Random (x, y) coordinates in 100x100 space

# Calculate distance between two cities
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Create distance matrix
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        distance_matrix[i][j] = distance(city_coordinates[i], city_coordinates[j])

# Route distance (total tour distance)
def route_distance(route):
    total_distance = 0
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        total_distance += distance_matrix[from_city][to_city]
    return total_distance

# Create initial population
def initial_population(pop_size, city_list):
    population = []
    for _ in range(pop_size):
        individual = random.sample(city_list, len(city_list))
        population.append(individual)
    return population

# Rank routes based on fitness (lower distance = better fitness)
def rank_routes(population):
    fitness_results = {}
    for i, route in enumerate(population):
        fitness_results[i] = 1 / route_distance(route)
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)

# Selection
def selection(ranked_routes, elite_size):
    selection_results = []

    # Keep elites
    for i in range(elite_size):
        selection_results.append(ranked_routes[i][0])

    # Roulette wheel selection
    df = pd.DataFrame(np.array(ranked_routes), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for _ in range(len(ranked_routes) - elite_size):
        pick = random.random() * 100
        for i in range(len(ranked_routes)):
            if pick <= df.iat[i, 3]:
                selection_results.append(ranked_routes[i][0])
                break

    return selection_results

# Mating pool
def mating_pool(population, selection_results):
    matingpool = []
    for i in selection_results:
        matingpool.append(population[i])
    return matingpool

# Crossover (Order Crossover)
def crossover(parent1, parent2):
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(0, len(parent1) - 1)

    if start > end:
        start, end = end, start

    child_p1 = parent1[start:end]
    child_p2 = [item for item in parent2 if item not in child_p1]

    child = child_p2[:start] + child_p1 + child_p2[start:]
    return child

# Crossover population
def crossover_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    # Elites go directly to next generation
    for i in range(elite_size):
        children.append(matingpool[i])

    for i in range(length):
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)

    return children

# Mutation (swap two cities)
def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(individual) - 1)
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual

# Mutate population
def mutate_population(population, mutation_rate):
    mutated_pop = []

    for ind in population:
        mutated_ind = mutate(ind, mutation_rate)
        mutated_pop.append(mutated_ind)

    return mutated_pop

# Next generation
def next_generation(current_gen, elite_size, mutation_rate):
    ranked = rank_routes(current_gen)
    selection_results = selection(ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = crossover_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen

# Main Genetic Algorithm

def genetic_algorithm(pop_size=100, elite_size=20, mutation_rate=0.01, generations=500):
    city_list = list(range(num_cities))
    population = initial_population(pop_size, city_list)
    print(f"Initial best distance: {1 / rank_routes(population)[0][1]:.2f}")

    for i in range(generations):
        population = next_generation(population, elite_size, mutation_rate)

    best_route_index = rank_routes(population)[0][0]
    best_route = population[best_route_index]
    print(f"Final best distance: {1 / rank_routes(population)[0][1]:.2f}")
    return best_route

# Run the GA
best_route = genetic_algorithm()
print("Best Route:", best_route)
