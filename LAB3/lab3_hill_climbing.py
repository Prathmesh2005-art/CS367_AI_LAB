from lab3_gen import generate_k_sat_problem
import random

# Definition of Node class representing a state (truth assignment)
class Node:
    def _init_(self, state):
        self.state = state


# Heuristic function 1 — counts number of satisfied clauses
def heuristic_value_1(clause, node):
    count = 0
    for curr_clause in clause:
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                count += 1
                break
            if literal < 0 and node.state[abs(literal) - 1] == 0:
                count += 1
                break
    return count


# Heuristic function 2 — counts total number of literals that evaluate to True
def heuristic_value_2(clause, node):
    count = 0
    for curr_clause in clause:
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                count += 1
            elif literal < 0 and node.state[abs(literal) - 1] == 0:
                count += 1
    return count


# Check if the current node satisfies all clauses
def check(clause, node):
    for curr_clause in clause:
        satisfied = False
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                satisfied = True
                break
            if literal < 0 and node.state[abs(literal) - 1] == 0:
                satisfied = True
                break
        if not satisfied:
            return False
    return True


# Generate the best successor (Hill Climbing step)
def gen_successor(node, clause):
    max_val = -1
    best_node = node

    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]  # Flip the bit
        new_node = Node(temp)
        val = heuristic_value_2(clause, new_node)
        if val > max_val:
            max_val = val
            best_node = new_node

    # If no improvement found (local maxima)
    if best_node.state == node.state:
        return None
    return best_node


# Hill Climbing Algorithm
def hill_climb(clause, k, m, n, max_iter=1000):
    node = Node([random.choice([0, 1]) for _ in range(n)])  # random start

    for i in range(max_iter):
        if check(clause, node):
            print("\n✅ Solution found!")
            print(f"Solution: {node.state}")
            print(f"Steps required: {i}")
            return True

        successor = gen_successor(node, clause)
        if successor is None:
            print("\n⚠ Local maxima reached (stuck).")
            return False

        node = successor

    print("\n❌ Max iterations reached without finding a solution.")
    return False


# Calculate Penetrance (success rate of solving)
def calculate_penetrance(num_instances, k, m, n):
    solved_count = 0
    for i in range(num_instances):
        print(f"\n🧩 Running instance {i+1}/{num_instances} ...")
        clauses = generate_k_sat_problem(k, m, n)
        if hill_climb(clauses, k, m, n):
            solved_count += 1
    penetrance = (solved_count / num_instances) * 100
    print(f"\n✅ Penetrance (success rate): {penetrance:.2f}%")
    return penetrance


# ------------------ Main Execution ------------------

if _name_ == "_main_":
    k, m, n = 3, 100, 100
    clause = generate_k_sat_problem(k, m, n)
    print("Generated Clause Example:\n", clause[:5], "...\n")

    # Test with 20 random SAT instances
    calculate_penetrance(20, 3, 50, 50)