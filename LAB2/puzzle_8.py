import numpy as np
from time import time
from random import randint


class Node:
    def _init_(self, parent, state, pcost, hcost):
        self.parent = parent
        self.state = state
        self.pcost = pcost  # Path cost
        self.hcost = hcost  # Heuristic cost
        self.cost = pcost + hcost  # Total cost

    def _hash_(self):
        return hash(tuple(self.state.flatten()))

    def _eq_(self, other):
        return np.array_equal(self.state, other.state)

    def _str_(self):
        return str(self.state)


class PriorityQueue:
    def _init_(self):
        self.queue = []

    def push(self, node):
        self.queue.append(node)

    def pop(self):
        # Pop node with minimum cost (A* priority)
        best_idx = min(range(len(self.queue)), key=lambda i: self.queue[i].cost)
        return self.queue.pop(best_idx)

    def is_empty(self):
        return len(self.queue) == 0

    def _len_(self):
        return len(self.queue)


class Environment:
    def _init_(self, depth, goal_state):
        self.goal_state = goal_state
        self.depth = depth
        self.start_state = self.generate_start_state()

    def generate_start_state(self):
        """Generate a random solvable start state by making random moves from the goal."""
        state = np.copy(self.goal_state)
        for _ in range(self.depth):
            next_states = self.get_next_states(state)
            choice = randint(0, len(next_states) - 1)
            new_state = next_states[choice]
            if not np.array_equal(new_state, state):
                state = new_state
        return state

    def get_start_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal_state

    def get_next_states(self, state):
        """Generate all valid next moves (up, down, left, right)."""
        i, j = np.where(state == "_")
        i, j = int(i[0]), int(j[0])
        moves = []
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # Up, Down, Right, Left

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_state = np.copy(state)
                new_state[i, j], new_state[ni, nj] = new_state[ni, nj], new_state[i, j]
                moves.append(new_state)
        return moves

    def reached_goal(self, state):
        return np.array_equal(state, self.goal_state)


class Agent:
    def _init_(self, env, heuristic):
        self.frontier = PriorityQueue()
        self.explored = set()
        self.env = env
        self.heuristic = heuristic
        self.goal_node = None

    def run(self):
        start_state = self.env.get_start_state()
        goal_state = self.env.get_goal_state()

        start_node = Node(None, start_state, 0, self.heuristic(start_state, goal_state))
        self.frontier.push(start_node)

        steps = 0
        while not self.frontier.is_empty():
            current = self.frontier.pop()
            if tuple(current.state.flatten()) in self.explored:
                continue

            self.explored.add(tuple(current.state.flatten()))

            if self.env.reached_goal(current.state):
                self.goal_node = current
                break

            for next_state in self.env.get_next_states(current.state):
                if tuple(next_state.flatten()) not in self.explored:
                    h = self.heuristic(next_state, goal_state)
                    node = Node(current, next_state, current.pcost + 1, h)
                    self.frontier.push(node)

            steps += 1

        return steps, self.solution_depth()

    def solution_depth(self):
        node = self.goal_node
        depth = 0
        while node is not None:
            node = node.parent
            depth += 1
        return depth

    def get_memory(self):
        # Rough estimate of memory used
        return (len(self.frontier) + len(self.explored)) * 56


# ---------- Heuristics ----------
def heuristic0(curr_state, goal_state):
    """Trivial heuristic (Uniform Cost Search)."""
    return 0


def heuristic1(curr_state, goal_state):
    """Count of misplaced tiles."""
    count = 0
    for i in range(3):
        for j in range(3):
            if curr_state[i, j] != goal_state[i, j] and curr_state[i, j] != "_":
                count += 1
    return count


def heuristic2(curr_state, goal_state):
    """Manhattan distance heuristic."""
    dist = 0
    for i in range(3):
        for j in range(3):
            val = curr_state[i, j]
            if val == "_":
                continue
            goal_i, goal_j = np.where(goal_state == val)
            dist += abs(goal_i[0] - i) + abs(goal_j[0] - j)
    return dist


# ---------- Main Experiment ----------
goal_state = np.array([[1, 2, 3], [8, "_", 4], [7, 6, 5]])
depths = np.arange(0, 501, 50)

times_taken = {}
memories = {}

print("Running A* on 8-puzzle with varying search depths...\n")

for depth in depths:
    avg_time = 0
    avg_mem = 0
    for _ in range(10):  # reduced for faster test; can use 50 for full test
        env = Environment(depth=depth, goal_state=goal_state)
        agent = Agent(env=env, heuristic=heuristic2)

        start_time = time()
        agent.run()
        end_time = time()

        avg_time += end_time - start_time
        avg_mem += agent.get_memory()

    avg_time /= 10
    avg_mem /= 10
    times_taken[depth] = avg_time
    memories[depth] = avg_mem
    print(f"Depth {depth:3d} | Avg Time: {avg_time:.5f}s | Avg Memory: {avg_mem:.2f} bytes")

print("\n✅ Experiment completed successfully.")