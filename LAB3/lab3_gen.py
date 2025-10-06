import random

def generate_k_sat_problem(k, m, n):
    """
    Generate a random k-SAT problem instance.

    Parameters:
        k (int): Number of literals per clause.
        m (int): Number of clauses.
        n (int): Number of distinct boolean variables.

    Returns:
        list[list[int]]: A list of clauses, where each clause is a list of integers.
                         Positive integers represent normal variables,
                         negative integers represent negated variables.
                         Example: [[1, -2, 3], [-1, 2, -3]]
    """
    clauses = []

    for _ in range(m):
        clause = set()  # Use set to avoid duplicate literals in a clause

        while len(clause) < k:
            var = random.randint(1, n)  # choose variable from 1..n
            is_negated = random.choice([True, False])
            literal = -var if is_negated else var
            clause.add(literal)

        # Convert set to sorted list (sorted by absolute variable index)
        clauses.append(sorted(list(clause), key=abs))

    return clauses


# Example usage
if _name_ == "_main_":
    k = 3   # 3 literals per clause
    m = 5   # 5 clauses
    n = 4   # 4 variables (x1..x4)
    formula = generate_k_sat_problem(k, m, n)
    print("Generated 3-SAT problem:")
    for i, clause in enumerate(formula, 1):
        print(f"Clause {i}: {clause}")