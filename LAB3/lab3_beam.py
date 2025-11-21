from lab3_gen import generate_k_sat_problem
import random


class Node:
    def _init_(self, state):
        self.state = state  # list of 0/1 assignments

    def copy(self):
        return Node(self.state.copy())


# Heuristics & Checks 
def heuristic_value_1(clauses, node):
    """
    Heuristic 1: Number of satisfied clauses (classic).
    A clause is satisfied if any literal in it is true under node.state.
    """
    satisfied = 0
    for clause in clauses:
        for lit in clause:
            if lit > 0 and node.state[lit - 1] == 1:
                satisfied += 1
                break
            if lit < 0 and node.state[abs(lit) - 1] == 0:
                satisfied += 1
                break
    return satisfied


def heuristic_value_2(clauses, node):
    """
    Heuristic 2: Count of satisfied literals across all clauses.
    (Strongly correlated with heuristic_value_1 but more granular.)
    """
    count = 0
    for clause in clauses:
        for lit in clause:
            if lit > 0 and node.state[lit - 1] == 1:
                count += 1
            elif lit < 0 and node.state[abs(lit) - 1] == 0:
                count += 1
    return count


def check_solution(clauses, node):
    """Return True if node satisfies all clauses."""
    for clause in clauses:
        sat = False
        for lit in clause:
            if lit > 0 and node.state[lit - 1] == 1:
                sat = True
                break
            if lit < 0 and node.state[abs(lit) - 1] == 0:
                sat = True
                break
        if not sat:
            return False
    return True


# Neighbor Generators 
def gen_successor_best_onebit(node, clauses):
    """
    Standard greedy neighbor: flip each bit and choose the neighbor
    that maximizes heuristic_value_1 (number of satisfied clauses).
    Return None if no neighbor improves over current node.
    """
    best_val = heuristic_value_1(clauses, node)
    best_node = None

    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]  # flip bit
        cand = Node(temp)
        val = heuristic_value_1(clauses, cand)
        if val > best_val:
            best_val = val
            best_node = cand

    return best_node  # None if no improvement


def generate_all_onebit_successors(node, clauses, beam_width=3):
    """
    Generate all 1-bit-flip neighbors, sort them by heuristic_value_1
    (or heuristic_value_2 if you prefer), and return the top beam_width.
    """
    successors = []
    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]
        successors.append(Node(temp))

    # sort by heuristic (more satisfied clauses better). Use hv1 for beam selection.
    successors.sort(key=lambda n: heuristic_value_1(clauses, n), reverse=True)

    # return the top-k nodes (beam)
    return successors[:beam_width]


def generate_random_multi_bit_successor(node, bits_to_flip=2):
    """Return a node created by flipping bits_to_flip distinct random bits."""
    n = len(node.state)
    indices = random.sample(range(n), bits_to_flip)
    temp = node.state.copy()
    for idx in indices:
        temp[idx] = 1 - temp[idx]
    return Node(temp)


# Hill Climbing (single restart) 
def hill_climb(clauses, start_node, max_iter=1000):
    """
    Greedy hill climb from start_node using single-bit best-improvement.
    Returns a node (best reached) or None if stuck immediately.
    """
    node = start_node.copy()
    for step in range(max_iter):
        if check_solution(clauses, node):
            return node
        next_node = gen_successor_best_onebit(node, clauses)
        if next_node is None:
            # stuck in local maxima
            return node
        node = next_node
    return node


# Beam Search 
def beam_search(clauses, n_vars, beam_width=3, max_iter=1000):
    """
    Beam search:
      - Start with beam_width random nodes (diversity).
      - At each iteration expand each beam node (all 1-bit neighbors),
        gather all candidates, keep top-k by heuristic_value_1.
      - Stop if any candidate solves the formula.
    Returns: solution Node if found, else None.
    """
    # initialize beam with random nodes
    beam = [Node([random.choice([0, 1]) for _ in range(n_vars)]) for _ in range(beam_width)]

    # early check
    for idx, n in enumerate(beam):
        if check_solution(clauses, n):
            # found immediately
            return n

    for iteration in range(max_iter):
        all_candidates = []
        for bnode in beam:
            # expand 1-bit neighbors
            candidates = generate_all_onebit_successors(bnode, clauses, beam_width=n_vars)
            # we can also add the original node (allow staying)
            candidates.append(bnode.copy())
            all_candidates.extend(candidates)

        # remove duplicates (by tuple(state))
        unique = {}
        for cand in all_candidates:
            key = tuple(cand.state)
            if key not in unique:
                unique[key] = cand

        # sort unique candidates by heuristic (use hv1)
        sorted_candidates = sorted(unique.values(), key=lambda n: heuristic_value_1(clauses, n), reverse=True)

        # keep top beam_width nodes
        beam = sorted_candidates[:beam_width]

        # check if any beam node satisfies clauses
        for b in beam:
            if check_solution(clauses, b):
                return b

        # optionally: if none candidate improved over previous beam (stagnation), we can try
        # random perturbation or restart — but for simplicity we continue.

    return None


# Penetrance / Experiment 
def calculate_penetrance(num_instances, k, m, n_vars, beam_width=3, max_iter=500):
    solved_count = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n_vars)
        sol = beam_search(clauses, n_vars, beam_width=beam_width, max_iter=max_iter)
        if sol is not None:
            solved_count += 1
    penetrance = (solved_count / num_instances) * 100.0
    return penetrance


# Example usage 
if _name_ == "_main_":
    # small sanity example (adjust params to taste)
    k = 3
    m = 25   # number of clauses
    n = 25   # number of variables
    trials = 20
    beam_w = 5
    print(f"Running {trials} instances of {k}-SAT (m={m}, n={n}) with beam_width={beam_w} ...")
    p = calculate_penetrance(trials, k, m, n, beam_width=beam_w, max_iter=300)

    print(f"Penetrance (success rate): {p:.2f}%")
