import heapq

# ----------------------------- Node Definition -----------------------------

class SolitaireNode:
    def __init__(self, state, parent=None, g=0, h=0, w1=1, w2=1):
        """
        Represents a node in the search tree for Peg Solitaire.
        state : 7x7 grid (list of lists)
        g      : Cost from start node
        h      : Heuristic estimate to goal
        f      : Total estimated cost (weighted)
        """
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = w1 * g + w2 * h

    def __lt__(self, other):
        return self.f < other.f


# ----------------------------- Helper Functions -----------------------------

def get_possible_moves(state):
    """
    Returns a list of valid moves in the current board configuration.
    Each move is a tuple: (start_row, start_col, end_row, end_col)
    """
    moves = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]      # Up, Down, Left, Right
    jumps = [(-1, 0), (1, 0), (0, -1), (0, 1)]           # The marble to be jumped over

    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':  # Found a marble
                for (dr, dc), (jr, jc) in zip(directions, jumps):
                    end_r, end_c = r + dr, c + dc
                    jump_r, jump_c = r + jr, c + jc

                    # Check bounds
                    if 0 <= end_r < 7 and 0 <= end_c < 7:
                        # Check valid jump: marble to jump over + empty destination
                        if state[jump_r][jump_c] == 'O' and state[end_r][end_c] == '0':
                            moves.append((r, c, end_r, end_c))

    return moves


def apply_move(state, move):
    """
    Apply a move to the current state and return the new resulting board.
    """
    new_state = [row[:] for row in state]  # Deep copy
    start_r, start_c, end_r, end_c = move
    jump_r, jump_c = (start_r + end_r) // 2, (start_c + end_c) // 2

    # Perform the move
    new_state[start_r][start_c] = '0'  # Empty start
    new_state[jump_r][jump_c] = '0'    # Remove jumped marble
    new_state[end_r][end_c] = 'O'      # Place marble at destination

    return new_state


# ----------------------------- Heuristic Functions -----------------------------

def heuristic_1(state):
    """Heuristic 1: Number of marbles remaining (lower is better)."""
    return sum(row.count('O') for row in state)


def heuristic_2(state):
    """Heuristic 2: Total Manhattan distance of marbles from center."""
    center = (3, 3)
    total_distance = 0
    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                total_distance += abs(r - center[0]) + abs(c - center[1])
    return total_distance


# ----------------------------- Search Algorithms -----------------------------

def best_first_search(initial_state, heuristic_func):
    """
    Greedy Best-First Search (uses only heuristic function h).
    """
    start_node = SolitaireNode(initial_state, h=heuristic_func(initial_state))
    open_list = []
    heapq.heappush(open_list, (start_node.h, start_node))
    visited = set()
    max_queue_size = 0

    while open_list:
        max_queue_size = max(max_queue_size, len(open_list))
        _, node = heapq.heappop(open_list)

        state_tuple = tuple(map(tuple, node.state))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        # Goal condition: Only one marble left
        if heuristic_func(node.state) == 1:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("\n--- Best-First Search ---")
            print(f"Max queue size: {max_queue_size}")
            print(f"Nodes visited: {len(visited)}")
            print(f"Path length: {len(path)}")
            return path[::-1]

        # Explore neighbors
        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            h = heuristic_func(new_state)
            new_node = SolitaireNode(new_state, parent=node, h=h)
            heapq.heappush(open_list, (new_node.h, new_node))

    return None


def a_star_search(initial_state, heuristic_func):
    """
    A* Search algorithm.
    """
    start_node = SolitaireNode(initial_state, h=heuristic_func(initial_state))
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited = set()
    max_queue_size = 0

    while open_list:
        max_queue_size = max(max_queue_size, len(open_list))
        _, node = heapq.heappop(open_list)

        state_tuple = tuple(map(tuple, node.state))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        # Goal condition: only one marble left
        if heuristic_func(node.state) == 1:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("\n--- A* Search ---")
            print(f"Max queue size: {max_queue_size}")
            print(f"Nodes visited: {len(visited)}")
            print(f"Path length: {len(path)}")
            return path[::-1]

        # Generate successors
        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            g = node.g + 1
            h = heuristic_func(new_state)
            new_node = SolitaireNode(new_state, parent=node, g=g, h=h)
            heapq.heappush(open_list, (new_node.f, new_node))

    return None


# ----------------------------- Main Execution -----------------------------

if __name__ == "__main__":
    start_state = [
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', '0', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['-', '-', 'O', 'O', 'O', '-', '-']
    ]

    moves = get_possible_moves(start_state)
    print("Initial possible moves:")
    for move in moves:
        print(f"Start: {move[0], move[1]} → End: {move[2], move[3]}")

    best_first_search(start_state, heuristic_2)
    a_star_search(start_state, heuristic_2)
