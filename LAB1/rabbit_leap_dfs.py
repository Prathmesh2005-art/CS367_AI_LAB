from collections import deque

# Check if the given state is valid
def is_valid(state):
    # Just ensure state length is 7 and there is exactly one empty spot
    if len(state) != 7 or state.count(-1) != 1:
        return False
    return True


# Swap helper function
def swap(state, i, j):
    new_state = list(state)
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return tuple(new_state)


# Generate possible successor states
def get_successors(state):
    successors = []
    empty = list(state).index(-1)
    moves = [-2, -1, 1, 2]  # Frogs can move or jump over one
    for move in moves:
        new_pos = empty + move
        if 0 <= new_pos < 7:
            # Left frog (0) moves right → positive move
            if move > 0 and state[new_pos] == 1:
                continue
            # Right frog (1) moves left → negative move
            if move < 0 and state[new_pos] == 0:
                continue
            new_state = swap(state, empty, new_pos)
            if is_valid(new_state):
                successors.append(new_state)
    return successors


# BFS search for shortest path
def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    count = 0
    max_size = 0

    while queue:
        max_size = max(max_size, len(queue))
        state, path = queue.popleft()  # Correct BFS behavior

        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        count += 1

        if state == goal_state:
            print(f"✅ Total nodes visited: {count}")
            print(f"📦 Max queue size: {max_size}")
            return path

        for successor in get_successors(state):
            queue.append((successor, path))
    return None


# Initial and goal states
start_state = (0, 0, 0, -1, 1, 1, 1)
goal_state = (1, 1, 1, -1, 0, 0, 0)

# Run BFS
solution = bfs(start_state, goal_state)

# Display result
if solution:
    print("\nSolution found!\n")
    print(f"Number of nodes in solution: {len(solution)}\n")
    for step in solution:
        print(step)
else:
    print("No solution found.")