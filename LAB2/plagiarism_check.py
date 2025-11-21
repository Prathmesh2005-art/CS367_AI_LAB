import heapq
from collections import defaultdict
import spacy
import string


# - File Reading - #
def read_file(file_path):
    """Read and return content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# - Sentence Extraction - #
def extract_sentences(text):
    """Extract and lowercase sentences using spaCy."""
    text = text.lower()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def remove_punctuation(sentence):
    """Remove punctuation and newlines from a sentence."""
    return sentence.translate(str.maketrans("", "", string.punctuation)).replace("\n", " ")


# - Text Processing - #
def process_text_file(file_path):
    """Read file, extract sentences, and clean punctuation."""
    content = read_file(file_path)
    sentences = extract_sentences(content)
    cleaned_sentences = [remove_punctuation(sentence) for sentence in sentences]
    return cleaned_sentences


# - Node Class - #
class Node:
    def _init_(self, state, parent=None, g=0, h=0, w1=1, w2=1):
        self.state = state
        self.parent = parent
        self.g = g  # cost from start
        self.h = h  # heuristic
        self.f = w1 * g + w2 * h

    def _lt_(self, other):
        return self.f < other.f


# - Global Docs - #
document1 = []
document2 = []


# - Utility Functions - #
def get_difference(index_doc1=None, index_doc2=None):
    """Compute difference in characters between two sentences."""
    sentence1 = document1[index_doc1] if index_doc1 is not None else ""
    sentence2 = document2[index_doc2] if index_doc2 is not None else ""

    chars1 = list(sentence1)
    chars2 = list(sentence2)

    char_count = defaultdict(int)
    for char in chars1:
        char_count[char] += 1

    difference = 0
    for char in chars2:
        if char_count[char] > 0:
            char_count[char] -= 1
        else:
            difference += 1

    # Length difference penalty
    difference += abs(len(chars1) - len(chars2))
    return difference


def distance(state, goal_state):
    """Estimate distance from current to goal state."""
    index_doc1, index_doc2, _ = state
    len_doc1, len_doc2, _ = goal_state

    dist = 0
    while index_doc1 < len_doc1 and index_doc2 < len_doc2:
        dist += get_difference(index_doc1, index_doc2)
        index_doc1 += 1
        index_doc2 += 1
    return dist


# - Edit Distance - #
def char_level_edit_distance(str1, str2):
    """Compute character-level edit distance (Levenshtein)."""
    n, m = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )
    return dp[n][m]


def edit_distance_cost(state):
    """Return the edit cost for a given move."""
    i, j, move = state
    if move == 0:
        return char_level_edit_distance(document1[i - 1], document2[j - 1])
    elif move == 1:
        return len(document2[j - 1])  # insertion
    elif move == 2:
        return len(document1[i - 1])  # deletion
    return 0


# - A* Successors - #
def get_successors(node):
    """Return next valid states from current node."""
    moves = [(1, 1, 0), (0, 1, 1), (1, 0, 2)]
    successors = []
    for dx, dy, move in moves:
        new_state = (node.state[0] + dx, node.state[1] + dy, move)
        successors.append(Node(new_state, node))
    return successors


# - A* Algorithm - #
def a_star(start_state, goal_state):
    """Run A* search algorithm."""
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited = set()
    explored = 0

    while open_list:
        _, node = heapq.heappop(open_list)
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        explored += 1

        if node.state[0] == goal_node.state[0] + 1 and node.state[1] == goal_node.state[1] + 1:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print(f"Total nodes explored: {explored}")
            return path[::-1]

        for successor in get_successors(node):
            if successor.state[0] <= goal_node.state[0] + 1 and successor.state[1] <= goal_node.state[1] + 1:
                successor.g = node.g + edit_distance_cost(successor.state)
                successor.h = distance(successor.state, goal_node.state)
                successor.f = successor.g + successor.h
                heapq.heappush(open_list, (successor.f, successor))

    print(f"⚠ No path found. Nodes explored: {explored}")
    return None


# - Document Alignment - #
def align_documents(states, start_state, goal_state):
    """Construct aligned document using computed path."""
    new_doc = []
    for st in states:
        move = st[-1]
        if st[:2] == start_state[:2]:
            continue
        if move == 0:
            new_doc.append(document1[st[0] - 1])
        elif move == 1:
            new_doc.append(document2[st[1] - 1])
        elif move == 2:
            continue

        if st[:2] == goal_state[:2]:
            print("🎯 Goal state reached")
    return new_doc


def word_count(sentence):
    return len(sentence.split())


# - Main Execution - #
if _name_ == "_main_":
    document1 = process_text_file("doc1.txt")
    document2 = process_text_file("doc2.txt")

    start_state = (0, 0, 0)
    goal_state = (len(document1) - 1, len(document2) - 1, 0)

    print(f"📘 Goal State: {goal_state}")
    result_path = a_star(start_state, goal_state)

    if result_path:
        aligned_doc = align_documents(result_path, start_state, goal_state)
        print("\n Aligned Document:")
        print(aligned_doc)

        print(f"\n Document1 ({len(document1)} sentences): {document1}")
        print(f"Document2 ({len(document2)} sentences): {document2}")

        for i in range(min(len(aligned_doc), len(document2))):
            dist = char_level_edit_distance(aligned_doc[i], document2[i])
            print(f"\n Sentence pair {i+1}:")
            print(f"Doc3: {aligned_doc[i]}")
            print(f"Doc2: {document2[i]}")

            print(f"Edit distance = {dist}")
