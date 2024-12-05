import numpy as np

def objective_function(x):
    """Objective function to minimize."""
    return x[0]**2 + x[1]**2

def constraint(x):
    """Constraint function."""
    return x[0] + x[1] - 2

def initialize_population(pop_size, bounds):
    """Generate the initial population."""
    lower, upper = bounds
    return np.random.uniform(lower, upper, (pop_size, len(lower)))

def evaluate_fitness(population, penalty_factor, epsilon):
    """Evaluate fitness with penalties for constraint violations."""
    fitness = []
    for individual in population:
        obj_val = objective_function(individual)
        constraint_val = max(0, abs(constraint(individual)) - epsilon)
        fitness.append(obj_val + penalty_factor * constraint_val)
    return np.array(fitness)

def select(population, fitness):
    """Selection using roulette wheel."""
    probabilities = 1 / (1 + fitness)  # Inverse fitness for minimization
    probabilities /= np.sum(probabilities)
    indices = np.random.choice(range(len(population)), size=len(population), p=probabilities)
    return population[indices]

def crossover(parent1, parent2, crossover_prob):
    """Perform crossover between two parents."""
    if np.random.rand() < crossover_prob:
        return (parent1 + parent2) / 2
    return parent1

def mutate(individual, mutation_prob, bounds):
    """Apply mutation to an individual."""
    if np.random.rand() < mutation_prob:
        lower, upper = bounds
        individual += np.random.uniform(-0.1, 0.1, size=individual.shape)
        individual = np.clip(individual, lower, upper)
    return individual

def SAGA_with_proportions(bounds, pop_size=200, generations=100, alpha=7, crossover_prob=0.65, mutation_prob=0.05, epsilon_init=0.1, epsilon_final=0.001):
    """SAGA with Three-Stage Evolution and Proportional Selection."""
    epsilon = epsilon_init
    population = initialize_population(pop_size, bounds)
    best_solution, best_fitness = None, float('inf')

    for generation in range(generations):
        # Calculate penalty factor
        feasible_count = sum(1 for ind in population if constraint(ind) <= 0)
        penalty_factor = np.exp(alpha * (1 - feasible_count / pop_size)) - 1

        # Split population into feasible and infeasible groups
        feasible = [ind for ind in population if constraint(ind) <= 0]
        infeasible = [ind for ind in population if constraint(ind) > 0]

        # Determine stage and apply proportions
        if generation < generations // 3:
            # Stage 1: 20% feasible, 80% infeasible
            num_feasible = int(0.2 * pop_size)
            num_infeasible = pop_size - num_feasible
        elif generation < 2 * generations // 3:
            # Stage 2: 50% feasible, 50% infeasible
            num_feasible = num_infeasible = pop_size // 2
        else:
            # Stage 3: 80% feasible, 20% infeasible
            num_feasible = int(0.8 * pop_size)
            num_infeasible = pop_size - num_feasible

        # Select individuals based on the proportions
        selected_feasible = np.random.choice(feasible, size=min(num_feasible, len(feasible)), replace=True)
        selected_infeasible = np.random.choice(infeasible, size=min(num_infeasible, len(infeasible)), replace=True)
        selected_population = np.concatenate((selected_feasible, selected_infeasible))

        # Evaluate fitness
        fitness = evaluate_fitness(selected_population, penalty_factor, epsilon)

        # Track the best solution
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_solution = selected_population[min_idx]
            best_fitness = fitness[min_idx]

        # Apply genetic operators
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else parent1
            offspring = crossover(parent1, parent2, crossover_prob)
            offspring = mutate(offspring, mutation_prob, bounds)
            new_population.append(offspring)
        population = np.array(new_population)

        # Adjust epsilon
        if generation % 5 == 0 and epsilon > epsilon_final:
            epsilon *= 0.8

        # Logging
        print(f"Generation {generation + 1}/{generations}, Best Fitness: {best_fitness:.4f}")

    return best_solution, best_fitness

# Define bounds and run the algorithm
bounds = ([-10, -10], [10, 10])  # Variable bounds for x and y
best_solution, best_fitness = SAGA_with_proportions(bounds)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
