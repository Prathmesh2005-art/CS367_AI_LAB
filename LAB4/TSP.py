import numpy as np
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Rajasthan Tourist Cities (Latitude, Longitude)
# ------------------------------------------------------------
locations = {
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Ajmer": (26.4499, 74.6399),
    "Jaisalmer": (26.9157, 70.9083),
    "Bikaner": (28.0229, 73.3119),
    "Mount Abu": (24.5926, 72.7156),
    "Pushkar": (26.4899, 74.5521),
    "Bharatpur": (27.2176, 77.4895),
    "Kota": (25.2138, 75.8648),
    "Chittorgarh": (24.8887, 74.6269),
    "Alwar": (27.5665, 76.6250),
    "Ranthambore": (26.0173, 76.5026),
    "Sariska": (27.3309, 76.4154),
    "Mandawa": (28.0524, 75.1416),
    "Dungarpur": (23.8430, 73.7142),
    "Bundi": (25.4305, 75.6499),
    "Sikar": (27.6094, 75.1399),
    "Nagaur": (27.2020, 73.7336),
    "Shekhawati": (27.6485, 75.5455),
}

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def euclidean_distance(coord1, coord2):
    """Compute Euclidean distance between two (lat, lon) coordinates."""
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


def compute_distance_matrix(locations):
    """Compute a full NxN distance matrix for the given locations."""
    cities = list(locations.keys())
    N = len(cities)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = euclidean_distance(locations[cities[i]], locations[cities[j]])
            D[j, i] = D[i, j]
    return cities, D


def tour_cost(tour, distance_matrix):
    """Compute total round-trip cost for a given tour."""
    cost = sum(distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    return cost + distance_matrix[tour[-1], tour[0]]


# ------------------------------------------------------------
# Simulated Annealing Algorithm
# ------------------------------------------------------------

def simulated_annealing(distance_matrix, max_iter=100000, temp_start=1000):
    """
    Simulated Annealing for the Traveling Salesman Problem.
    - distance_matrix: pairwise city distances
    - max_iter: maximum iterations
    - temp_start: initial temperature
    """
    N = len(distance_matrix)
    current_tour = random.sample(range(N), N)      # Initial random tour
    current_cost = tour_cost(current_tour, distance_matrix)

    best_tour = current_tour
    best_cost = current_cost
    cost_history = [current_cost]

    for iteration in range(1, max_iter + 1):
        # Create neighbor by reversing a segment (2-opt move)
        i, j = sorted(random.sample(range(N), 2))
        new_tour = current_tour[:i] + current_tour[i:j + 1][::-1] + current_tour[j + 1:]
        new_cost = tour_cost(new_tour, distance_matrix)

        delta_cost = new_cost - current_cost
        temperature = temp_start / iteration
        acceptance_prob = np.exp(-delta_cost / temperature) if delta_cost > 0 else 1

        # Accept new state if better or by probability
        if delta_cost < 0 or random.random() < acceptance_prob:
            current_tour = new_tour
            current_cost = new_cost

        # Track best solution
        if current_cost < best_cost:
            best_tour = current_tour
            best_cost = current_cost

        cost_history.append(best_cost)

    return best_tour, best_cost, cost_history


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def plot_results(cities, locations, best_tour, cost_history):
    """Plot optimized TSP route and cost evolution."""
    plt.figure(figsize=(12, 5))

    # ---- Plot optimized tour ----
    plt.subplot(1, 2, 1)
    tour_coords = np.array([locations[cities[i]] for i in best_tour] +
                           [locations[cities[best_tour[0]]]])
    plt.plot(tour_coords[:, 1], tour_coords[:, 0], "o-", lw=2, color="teal")
    for i, city in enumerate(best_tour):
        plt.text(tour_coords[i, 1], tour_coords[i, 0], cities[city], fontsize=9)
    plt.title("Optimized Rajasthan Tour")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # ---- Plot cost convergence ----
    plt.subplot(1, 2, 2)
    plt.plot(cost_history, color="darkorange")
    plt.title("Tour Cost Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Cost")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------

if __name__ == "__main__":
    cities, D = compute_distance_matrix(locations)
    best_tour, best_cost, cost_history = simulated_annealing(D)

    print("\n Best Tour Sequence:")
    print(" → ".join(cities[i] for i in best_tour))
    print(f"\n  Best Tour Cost: {best_cost:.4f}")

    plot_results(cities, locations, best_tour, cost_history)

