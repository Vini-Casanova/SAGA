# Código ajustado para compatibilidade no Windows
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# SAGA Implementation for f1 Benchmark

# f1 Benchmark Problem: Objective function and constraint
def f1_objective(x):
    return x[0]**2 + x[1]**2

def f1_constraint(x):
    return x[0] + x[1] - 2

# Parameters specific to SAGA and the f1 benchmark
pop_size = 200  # Population size
num_generations = 100  # Number of generations
alpha = 7  # Penalty function parameter
crossover_prob = 0.65
mutation_prob = 0.05
initial_epsilon = 0.1
final_epsilon = 0.001
num_runs = 30  # Number of independent runs

# Bounds for variables in f1 benchmark
bounds = ([-10, -10], [10, 10])

# Initialize population within bounds
def initialize_population(size, bounds):
    lower, upper = bounds
    return np.random.uniform(lower, upper, (size, len(lower)))

# Fitness calculation with penalty
def calculate_fitness(individual, penalty_factor, epsilon):
    obj_value = f1_objective(individual)
    constraint_value = max(0, abs(f1_constraint(individual)) - epsilon)
    return obj_value + penalty_factor * constraint_value

# Selection using roulette wheel method
def select(population, fitness_values):
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        fitness_values -= min_fitness  # Normalize to positive values
    probabilities = fitness_values / np.sum(fitness_values)
    indices = np.random.choice(range(len(population)), size=len(population), p=probabilities)
    return population[indices]

# Crossover operation
def crossover(parent1, parent2):
    if np.random.rand() < crossover_prob:
        return (parent1 + parent2) / 2, (parent1 + parent2) / 2
    return parent1, parent2

# Mutation operation
def mutate(individual, bounds, epsilon):
    lower, upper = bounds
    mutated = individual.copy()
    if np.random.rand() < mutation_prob:
        mutated += np.random.uniform(-epsilon, epsilon, individual.shape)
        mutated = np.clip(mutated, lower, upper)
    return mutated

# SAGA Algorithm with Fitness Reporting
def SAGA_f1_with_reporting(bounds, pop_size, num_generations):
    epsilon = initial_epsilon
    lower, upper = bounds
    population = initialize_population(pop_size, bounds)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []  # Collect best fitness per generation

    for generation in range(num_generations):
        penalty_factor = np.exp(alpha * (1 - np.count_nonzero(population) / pop_size)) - 1
        fitness_values = np.array([calculate_fitness(ind, penalty_factor, epsilon) for ind in population])

        # Track the best solution and log fitness
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = population[current_best_idx]

        fitness_history.append(best_fitness)  # Log the best fitness of the generation

        # Selection
        selected_population = select(population, fitness_values)

        # Generate new population
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else parent1
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, bounds, epsilon)
            offspring2 = mutate(offspring2, bounds, epsilon)
            new_population.extend([offspring1, offspring2])

        population = np.array(new_population)

        # Adjust epsilon dynamically
        if generation % 5 == 0 and epsilon > final_epsilon:
            epsilon *= 0.8

    return best_solution, best_fitness, fitness_history

# Perform 30 runs for faithful reproduction
results = []
all_histories = []

for run in range(num_runs):
    best_solution, best_fitness, fitness_history = SAGA_f1_with_reporting(bounds, pop_size, num_generations)
    results.append(best_fitness)
    all_histories.append(fitness_history)
    print(f"Run {run + 1}/{num_runs}: Best Fitness = {best_fitness}")

# Calculate statistics across runs
mean_fitness = np.mean(results)
std_fitness = np.std(results)

# Prepare results DataFrame for reporting
detailed_results_df = pd.DataFrame({
    'Run': range(1, num_runs + 1),
    'Best Fitness': results
})

# Aggregate fitness progression for all runs
aggregated_history = pd.DataFrame(all_histories).mean(axis=0)
history_df = pd.DataFrame({
    'Generation': range(1, len(aggregated_history) + 1),
    'Mean Fitness': aggregated_history
})

# Save detailed results and history
detailed_results_df.to_csv('SAGA_f1_detailed_results.csv', index=False)
history_df.to_csv('SAGA_f1_fitness_history.csv', index=False)

# Plot average fitness progression across generations
plt.figure(figsize=(10, 6))
plt.plot(history_df['Generation'], history_df['Mean Fitness'], marker='o')
plt.title("Average Fitness Progression Across 30 Runs for f1 Benchmark")
plt.xlabel("Generation")
plt.ylabel("Mean Fitness")
plt.grid(True)
plt.show()

# Generate boxplot for best fitness across runs
plt.figure(figsize=(8, 6))
plt.boxplot(results, vert=True, patch_artist=True)
plt.title("Distribution of Best Fitness Across 30 Runs for f1 Benchmark")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.show()

# Display results summary
print(f"Mean Best Fitness over 30 runs: {mean_fitness}")
print(f"Standard Deviation of Best Fitness: {std_fitness}")
