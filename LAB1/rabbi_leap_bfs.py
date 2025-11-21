from collections import deque

# Check if a state configuration is valid
def is_valid(state):
    empty = state.index(-1)
    
    # Goal state is always valid
    if state == (1, 1, 1, -1, 0, 0, 0):
        return True

    # Invalid configurations (frogs face-to-face without empty space)
    if empty == 0 and state[1] == 0 and state[2] == 0:
        return False
    if empty == 6 and state[5] == 1 and state[4] == 1:
        return False
    if empty == 1 and state[0] == 1 and state[2] == 0 and state[3] == 0:
        return False
    if empty == 5 and state[6] == 0 and state[4] == 1 and state[3] == 1:
        return False
    if state[empty - 1] == 1 and state[empty - 2] == 1 and state[empty + 1] == 0 and state[empty + 2] == 0:
        return False
    
    return True


# Swap two positions to create a new state
def swap(state, i, j):
    new_state = list(state)
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return tuple(new_state)


# Generate all valid successor states from current state
def get_successors(state):
    successors = []
    empty = state.index(-1)
    moves = [-2, -1, 1, 2]

    for move in moves:
        new_pos = empty + move
        if 0 <= new_pos < len(state):
            # Rule: right-moving frogs (1) move only to the right
            # and left-moving frogs (0) move only to the left
            if move > 0 and state[new_pos] == 1:  
                new_state = swap(state, empty, new_pos)
                if is_valid(new_state):
                    successors.append(new_state)

            elif move < 0 and state[new_pos] == 0:  
                new_state = swap(state, empty, new_pos)
                if is_valid(new_state):
                    successors.append(new_state)

    return successors


# Breadth-First Search algorithm
def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    nodes_visited = 0
    max_queue_size = 0

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        state, path = queue.popleft()

        if state in visited:
            continue

        visited.add(state)
        nodes_visited += 1
        path = path + [state]

        # Goal check
        if state == goal_state:
            print(f"\n Solution Found!")
            print(f"Total Nodes Visited: {nodes_visited}")
            print(f"Maximum Queue Size: {max_queue_size}")
            return path

        # Explore successors
        for successor in get_successors(state):
            queue.append((successor, path))

    return None


# --- Main Execution ---
start_state = (0, 0, 0, -1, 1, 1, 1)
goal_state = (1, 1, 1, -1, 0, 0, 0)

solution = bfs(start_state, goal_state)

if solution:
    print(f"\nNumber of States in Solution Path: {len(solution)}")
    for step in solution:
        print(step)
else:
    print("\nNo solution found.")

