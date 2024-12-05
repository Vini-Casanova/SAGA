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
pop_size = 200
num_generations = 100
alpha = 7
crossover_prob = 0.65
mutation_prob = 0.05
initial_epsilon = 0.1
final_epsilon = 0.001
num_runs = 30

bounds = ([-10, -10], [10, 10])  # Bounds for variables

# Initialize population within bounds
def initialize_population(size, bounds):
    lower, upper = bounds
    return np.random.uniform(lower, upper, (size, len(lower)))

# Fitness calculation with penalty
def calculate_fitness(individual, penalty_factor, epsilon):
    obj_value = f1_objective(individual)
    constraint_value = max(0, f1_constraint(individual) - epsilon)
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

# SAGA Algorithm with Reporting
def SAGA_f1_with_reporting(bounds, pop_size, num_generations):
    epsilon = initial_epsilon
    population = initialize_population(pop_size, bounds)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []
    violation_history = []

    for generation in range(num_generations):
        penalty_factor = np.exp(alpha * (1 - np.count_nonzero(population) / pop_size)) - 1
        fitness_values = np.array([calculate_fitness(ind, penalty_factor, epsilon) for ind in population])
        
        # Track constraint violations
        violations = [max(0, f1_constraint(ind) - epsilon) for ind in population]
        avg_violation = np.mean(violations)
        violation_history.append(avg_violation)

        # Track the best solution and log fitness
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = population[current_best_idx]

        fitness_history.append(best_fitness)

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

    return best_solution, best_fitness, fitness_history, violation_history

# Perform 30 runs and aggregate results
results = []
all_histories = []
all_violations = []

for run in range(num_runs):
    best_solution, best_fitness, fitness_history, violation_history = SAGA_f1_with_reporting(bounds, pop_size, num_generations)
    results.append(best_fitness)
    all_histories.append(fitness_history)
    all_violations.append(violation_history)
    print(f"Run {run + 1}/{num_runs}: Best Fitness = {best_fitness}")

# Aggregate results for visualization
aggregated_history = np.mean(all_histories, axis=0)
aggregated_violations = np.mean(all_violations, axis=0)

# Contour Plot for f1 Benchmark
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
constraint_line = 2 - x  # y = 2 - x

plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=np.logspace(0, 2.5, 15), cmap="rainbow")
plt.colorbar(contour, label="f(x, y)")
plt.plot(x, constraint_line, 'k-', linewidth=2, label="Constraint: x + y = 2")
plt.scatter([1], [1], color="red", label="Optimal Solution (1, 1)", s=100, edgecolor='black')
plt.title("f1 Benchmark: Objective Function and Constraint")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.grid(True)
plt.legend()
plt.show()

# Plot Objective Value Across Generations
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(aggregated_history) + 1), aggregated_history, label="Objective of f1", color="red")
plt.title("Objective of f1 Across Generations")
plt.xlabel("Generations")
plt.ylabel("Objective Value")
plt.grid(True)
plt.legend()
plt.show()

# Plot Constraint Violation Across Generations
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(aggregated_violations) + 1), aggregated_violations, label="Constraint Violation of f1", color="blue")
plt.title("Constraint Violation Across Generations")
plt.xlabel("Generations")
plt.ylabel("Constraint Violation")
plt.grid(True)
plt.legend()
plt.show()

# Combined Objective and Constraint Violation Plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Objective Plot
axs[0].plot(range(1, len(aggregated_history) + 1), aggregated_history, label="Objective of f1", color="red")
axs[0].set_title("(a) Objective of f1")
axs[0].set_xlabel("Generations")
axs[0].set_ylabel("Objective Value")
axs[0].grid(True)
axs[0].legend()

# Constraint Violation Plot
axs[1].plot(range(1, len(aggregated_violations) + 1), aggregated_violations, label="Constraint Violation of f1", color="blue")
axs[1].set_title("(b) Constraint Violation of f1")
axs[1].set_xlabel("Generations")
axs[1].set_ylabel("Constraint Violation")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

# Print Final Results
mean_fitness = np.mean(results)
std_fitness = np.std(results)
print(f"Mean Best Fitness: {mean_fitness}")
print(f"Standard Deviation of Best Fitness: {std_fitness}")
