from lab3_gen import generate_k_sat_problem
import random


# Node Definition 
class Node:
    def _init_(self, state):
        self.state = state  # binary assignment list [0,1,1,0,...]


# Heuristic Functions 
def heuristic_value_1(clauses, node):
    """Heuristic 1: Count number of satisfied clauses."""
    count = 0
    for curr_clause in clauses:
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                count += 1
                break
            elif literal < 0 and node.state[abs(literal) - 1] == 0:
                count += 1
                break
    return count


def heuristic_value_2(clauses, node):
    """Heuristic 2: Count total number of literals satisfied across all clauses."""
    count = 0
    for curr_clause in clauses:
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                count += 1
            elif literal < 0 and node.state[abs(literal) - 1] == 0:
                count += 1
    return count


#  Solution Check 
def check_solution(clauses, node):
    """Return True if node satisfies all clauses."""
    if node is None:
        return False
    satisfied = 0
    for curr_clause in clauses:
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                satisfied += 1
                break
            elif literal < 0 and node.state[abs(literal) - 1] == 0:
                satisfied += 1
                break
    return satisfied == len(clauses)


# Successor Generators 
def gen_1(node, clauses):
    """Generate successor by flipping one bit at a time (standard hill climb)."""
    best_value = -1
    best_node = node

    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]  # Flip bit
        new_node = Node(temp)
        val = heuristic_value_2(clauses, new_node)
        if val > best_value:
            best_value = val
            best_node = new_node

    # If no improvement, we are at local maxima
    if best_node.state == node.state:
        print("Local maxima reached (gen_1)")
        return None
    return best_node


def gen_2(node, clauses, num_neighbors=10):
    """Generate random successors by flipping one or two bits."""
    best_value = -1
    best_node = node

    for _ in range(num_neighbors):
        temp = node.state.copy()
        num_bits_to_flip = random.choice([1, 2])
        bits_to_flip = random.sample(range(len(node.state)), num_bits_to_flip)

        for bit in bits_to_flip:
            temp[bit] = 1 - temp[bit]

        new_node = Node(temp)
        val = heuristic_value_2(clauses, new_node)
        if val > best_value:
            best_value = val
            best_node = new_node

    if best_node.state == node.state:
        print("Local maxima reached (gen_2)")
        return None
    return best_node


def gen_3(node, clauses, num_neighbors=10):
    """Generate random successors by flipping 1, 2, or 3 bits."""
    best_value = -1
    best_node = node

    for _ in range(num_neighbors):
        temp = node.state.copy()
        num_bits_to_flip = random.choice([1, 2, 3])
        bits_to_flip = random.sample(range(len(node.state)), num_bits_to_flip)

        for bit in bits_to_flip:
            temp[bit] = 1 - temp[bit]

        new_node = Node(temp)
        val = heuristic_value_2(clauses, new_node)
        if val > best_value:
            best_value = val
            best_node = new_node

    if best_node.state == node.state:
        print("Local maxima reached (gen_3)")
        return None
    return best_node


# Hill Climb 
def hill_climb(clauses, node, gen_func, max_iter=1000):
    """Perform hill climbing using specified generator."""
    prev_node = node
    for i in range(max_iter):
        if check_solution(clauses, node):
            print("Solution found")
            print(f"Clause set: {clauses}")
            print(f"Solution: {node.state}")
            print(f"Steps required: {i}")
            return node

        temp_node = gen_func(node, clauses)
        if temp_node is None:
            print("⚠ Local maxima reached.")
            print(f"Best node so far: {node.state}")
            return node

        prev_node = node
        node = temp_node
    return node


# Variable Generation Node 
def vgn(clauses, k, m, n):
    """Run hill climbing with progressively more powerful neighbor generators."""
    node = Node([random.choice([0, 1]) for _ in range(n)])

    # First generator
    print("\n Running gen_1 ...")
    node = hill_climb(clauses, node, gen_1)
    if check_solution(clauses, node):
        print(" Found solution using gen_1")
        return True

    # Second generator
    print("\n⚙ Running gen_2 ...")
    node = hill_climb(clauses, node, gen_2)
    if check_solution(clauses, node):
        print(" Found solution using gen_2")
        return True

    # Third generator
    print("\n Running gen_3 ...")
    node = hill_climb(clauses, node, gen_3)
    if check_solution(clauses, node):
        print(" Found solution using gen_3")
        return True

    print(" No satisfying assignment found.")
    return False


# Penetrance Calculation 
def calculate_penetrance(num_instances, k, m, n):
    """Calculate the success rate of solving k-SAT using hill climbing."""
    solved_count = 0

    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        if vgn(clauses, k, m, n):
            solved_count += 1

    penetrance = (solved_count / num_instances) * 100
    print(f"\n Penetrance (Success Rate): {penetrance:.2f}%")
    return penetrance


# ----------------------------- Main ----------------------------- #
if _name_ == "_main_":
    # Example: k=3 (3-SAT), m=10 clauses, n=10 variables

    calculate_penetrance(num_instances=20, k=3, m=10, n=10)
