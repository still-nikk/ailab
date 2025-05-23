########## DFS ##########
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def dfs_recursive(edges, start_node):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    dfs_edges = []
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_edges.append((node, neighbor))
                dfs(neighbor)
    
    dfs(start_node)
    return dfs_edges

def dfs_iterative(edges, start_node):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    dfs_edges = []
    stack = [start_node]
    visited.add(start_node)
    
    while stack:
        current = stack.pop()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs_edges.append((current, neighbor))
                stack.append(neighbor)
    
    return dfs_edges

def plot_graph(edges, user_title, start_node):
    graph = nx.Graph()
    graph.add_edges_from(edges)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)  # Layout for node positioning
    node_colors = ["red" if node == start_node else "lightblue" for node in graph.nodes()]
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10, edge_color="gray")
    plt.title(user_title)
    plt.show()

edge_list = pd.read_csv('./graph-edges.csv')
edges = list(zip(edge_list['node'], edge_list['neighbor']))

dfs_edges_recursive = dfs_recursive(edges, start_node=7)
dfs_edges_iterative = dfs_iterative(edges, start_node=7)

print("Recursive DFS Tree Edges:", dfs_edges_recursive)
print("Iterative DFS Tree Edges:", dfs_edges_iterative)

plot_graph(edges, "Original Graph", start_node=7)
plot_graph(dfs_edges_recursive, "Recursive DFS Tree", start_node=7)
plot_graph(dfs_edges_iterative, "Iterative DFS Tree", start_node=7)




########## BFS ###########
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def bfs_tree(edges, start_node):
    # Create adjacency list with weights
    graph = {}
    for u, v, w in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append((v, w))
        graph[v].append((u, w))  # Assuming undirected graph

    visited = set()
    bfs_edges = []

    def bfs(node):
        queue = deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            for neighbor, weight in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    bfs_edges.append((current, neighbor, weight))
                    queue.append(neighbor)

    bfs(start_node)
    return bfs_edges    

def plot_graph(edges, user_title, start_node=None):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)  # Ensure weights are added
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)  # Positions for nodes
    node_colors = ["red" if node == start_node else "lightblue" for node in graph.nodes()]
    
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10, edge_color="gray")

    # Draw edge weights
    edge_labels = {(u, v): w for u, v, w in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(user_title)
    plt.show()

# Load edge list from CSV
edge_list = pd.read_csv('graph-edges.csv')
    
edges = list(zip(edge_list['node'], edge_list['neighbor'], edge_list['weights']))

# Compute BFS tree edges
start_node = 0  # Assuming starting node is 0
bfs_edges = bfs_tree(edges, start_node=start_node)

print("BFS Tree Edges:", bfs_edges)

# Plot the graphs
plot_graph(edges, "Original Graph")
plot_graph(bfs_edges, "BFS Tree", start_node=start_node)




########## Best First Search ##########
import networkx as nx 
import pandas as pd
import heapq
import matplotlib.pyplot as plt

def load_graph_from_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if not all(col in df.columns for col in ["source", "target", "weight"]):
        raise ValueError("CSV file must contain 'source', 'target', and 'weight' columns.")

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(str(row["source"]), str(row["target"]), weight=row["weight"])
    
    return G

def load_heuristic_from_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if not all(col in df.columns for col in ["node", "dist"]):
        raise ValueError("CSV file must contain 'node' and 'dist' columns.")

    return {str(row["node"]): row["dist"] for _, row in df.iterrows()}

def best_first_search(graph, start, goal, heuristic):
    op = []
    heapq.heappush(op, (heuristic[start], start, []))

    visited = set()

    while op:
        f, current, path = heapq.heappop(op)

        if current in visited:
            continue
        path = path + [current]

        if current == goal:
            return path

        visited.add(current)

        for neighbor in graph.neighbors(current):
            heapq.heappush(op, (heuristic[neighbor], neighbor, path))

    return None

def draw_graph(graph, path):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=5)
    labels = nx.get_edge_attributes(graph, 'weight')
    
    # Draw all edges in gray
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, label_pos=0.3)
    
    # Highlight the path in red
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2.5)
    
    plt.show()

graph = load_graph_from_csv("graph.csv")
heuristic = load_heuristic_from_csv("heuristic.csv")

start_node = "A"
goal_node = "L"
path = best_first_search(graph, start_node, goal_node, heuristic)

if path:
    print(f"Path from {start_node} to {goal_node} using Best First Search: {' -> '.join(path)}")
else:
    print("No path found.")

draw_graph(graph, path)




########## A* Search ##########
# A* Algorithm Full Version (CSV and Manual Input + Directed and Undirected Graphs)
import pandas as pd
import networkx as nx
import heapq
import matplotlib.pyplot as plt

# ---------------------------
# Choose the mode here
USE_CSV = True         # Set True for CSV input, False for manual user input
IS_DIRECTED = True     # Set True for Directed Graph, False for Undirected Graph
# ---------------------------

# Load Graph from CSV
def load_graph_from_csv(file_path, directed=True):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if not all(col in df.columns for col in ["source", "target", "weight"]):
        raise ValueError("CSV must have 'source', 'target', and 'weight' columns.")

    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row["source"]), str(row["target"]), weight=row["weight"])

    return G

# Load Heuristic from CSV
def load_heuristic_from_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if not all(col in df.columns for col in ["node", "dist"]):
        raise ValueError("CSV must have 'node' and 'dist' columns.")

    return {str(row["node"]): row["dist"] for _, row in df.iterrows()}

# Manual input Graph
def load_graph_from_user(directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    n = int(input("Enter number of edges: "))
    print("Enter edges in format: source target weight")

    for _ in range(n):
        u, v, w = input().split()
        G.add_edge(u, v, weight=float(w))
    return G

# Manual input Heuristic
def load_heuristic_from_user():
    heuristic = {}
    n = int(input("Enter number of heuristic entries: "))
    print("Enter heuristics in format: node heuristic_value")

    for _ in range(n):
        node, h = input().split()
        heuristic[node] = float(h)
    return heuristic

# A* Algorithm
def a_star(graph, start, goal, heuristic):
    op = []
    heapq.heappush(op, (0 + heuristic[start], 0, start, []))

    visited = set()

    while op:
        f, g, current, path = heapq.heappop(op)

        if current in visited:
            continue
        path = path + [current]

        if current == goal:
            return path, g

        visited.add(current)

        for neighbor in graph.neighbors(current):
            edge_weight = graph[current][neighbor]['weight']
            new_g = g + edge_weight
            new_f = new_g + heuristic[neighbor]
            heapq.heappush(op, (new_f, new_g, neighbor, path))

    return None, float('inf')

# Draw Graph and Path
def draw_graph(graph, path, title, color):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=0.5)
    labels = nx.get_edge_attributes(graph, 'weight')

    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, label_pos=0.3)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=color, width=3)

    plt.title(title)
    plt.show()

# Main Execution
if USE_CSV:
    graph = load_graph_from_csv("graph.csv", directed=IS_DIRECTED)
    heuristic = load_heuristic_from_csv("heuristic.csv")
else:
    graph = load_graph_from_user(directed=IS_DIRECTED)
    heuristic = load_heuristic_from_user()

start_node = input("Enter the start node: ")
goal_node = input("Enter the goal node: ")

path, cost = a_star(graph, start_node, goal_node, heuristic)

# Print Results
if path:
    print(f"Path from {start_node} to {goal_node}: {' -> '.join(path)} with total cost {cost}")
    draw_graph(graph, path, "A* Shortest Path", color="green")
else:
    print("No path found!")



# ---------------------------
# Viva Questions (Paste below as comments)

"""
A* Algorithm Viva Preparation Notes:

1. What is A* Algorithm?
- A* is a best-first search algorithm that finds the shortest path from a start node to a goal node using cost and heuristic estimates.

2. What is the formula for A* evaluation function?
- f(n) = g(n) + h(n)
  where,
  g(n) = cost from start to current node
  h(n) = estimated cost from current node to goal

3. What is heuristic in A*?
- A heuristic is an estimate of the cost from a node to the goal. It helps guide the search efficiently.

4. What happens if heuristic is zero?
- A* behaves like Dijkstra’s algorithm (pure uniform-cost search).

5. What properties make a heuristic good?
- Admissibility: It never overestimates the true cost.
- Consistency (Monotonicity): The estimated cost is always less than or equal to the estimated cost from a neighbor plus the cost to reach that neighbor.

6. Difference between directed and undirected graph in A*?
- Directed: edges have direction (A->B is not same as B->A).
- Undirected: edges work both ways (A->B and B->A are same).

7. Why use heapq (priority queue) in A*?
- To always expand the node with the lowest f(n) value first, ensuring optimal search.

8. If heuristic is very bad (non-admissible), what happens?
- A* might find a suboptimal path or behave inefficiently.

9. Why are we using networkx?
- For graph representation, easy edge and node management, and built-in methods.

10. Why spring_layout() is used in plotting?
- To automatically adjust nodes so the graph looks clean and readable.

11. Dealing with CSV vs Manual Input?
- CSV helps for bigger graphs and automated testing.
- Manual input is good for small examples and demos.

"""
# ---------------------------




########## Fuzzy ###########
import numpy as np

# Define fuzzy set operations


def fuzzy_union(A, B):
    """Union of two fuzzy sets A and B"""
    return np.maximum(A, B)


def fuzzy_intersection(A, B):
    """Intersection of two fuzzy sets A and B"""
    return np.minimum(A, B)


def fuzzy_complement(A):
    """Complement of a fuzzy set A"""
    return 1 - A


def demonstrate_de_morgan(A, B):
    """Demonstrate De Morgan's Laws"""
    complement_union = fuzzy_complement(fuzzy_union(A, B))
    complement_intersection = fuzzy_complement(fuzzy_intersection(A, B))

    # De Morgan's Law: Complement of Union = Complement of A intersect Complement of B
    law_1 = np.allclose(
        complement_union, fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))
    )

    # De Morgan's Law: Complement of Intersection = Complement of A union Complement of B
    law_2 = np.allclose(
        complement_intersection, fuzzy_union(fuzzy_complement(A), fuzzy_complement(B))
    )

    return complement_union, complement_intersection, law_1, law_2


# Fuzzy sets for demonstration
A = np.array([0.1, 0.4, 0.7, 0.8, 1.0])  # Fuzzy set A
B = np.array([0.3, 0.6, 0.2, 0.9, 0.5])  # Fuzzy set B
C = np.array([0.5, 0.2, 0.4, 0.6, 0.9])  # Fuzzy set C

# Demonstrating the fuzzy set operations

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)
print("Fuzzy Set C:", C)

# Union
print("\nUnion of A and B:", fuzzy_union(A, B))

# Intersection
print("\nIntersection of A and B:", fuzzy_intersection(A, B))

# Complement of A
print("\nComplement of A:", fuzzy_complement(A))

# De Morgan's Law Demonstration
complement_union, complement_intersection, law_1, law_2 = demonstrate_de_morgan(A, B)
print("\nComplement of Union A U B:", complement_union)
print("Complement of Intersection A ∩ B:", complement_intersection)
print("\nDe Morgan's Law (Complement of Union):", "Valid" if law_1 else "Invalid")
print("De Morgan's Law (Complement of Intersection):", "Valid" if law_2 else "Invalid")


# =============================================
#              FUZZY LOGIC - VIVA NOTES
# =============================================

# ----------- CRISP SET VS FUZZY SET -----------
# Crisp Set:
# - Membership is binary (0 or 1)
# - Elements either belong or do not belong
# - Example: A = {1, 2, 3}

# Fuzzy Set:
# - Membership is a value in [0, 1]
# - Elements can partially belong to a set
# - Example: A = {(1, 0.2), (2, 0.5), (3, 1.0)}

# ----------- FUZZY SET OPERATIONS -------------

# 1. Union (A ∪ B)
#    μ_A∪B(x) = max(μ_A(x), μ_B(x))
#    Meaning: Highest degree of belonging in either set

# 2. Intersection (A ∩ B)
#    μ_A∩B(x) = min(μ_A(x), μ_B(x))
#    Meaning: Commonality between the sets

# 3. Complement (A')
#    μ_A'(x) = 1 - μ_A(x)
#    Meaning: Degree to which x does not belong to A

# ----------- DE MORGAN’S LAWS -----------------

# 1. (A ∪ B)' = A' ∩ B'
# 2. (A ∩ B)' = A' ∪ B'

# Note: In fuzzy logic, these laws hold approximately due to floating-point math

# ------------ VIVA QUESTIONS ------------------

# Q1: What is a fuzzy set?
# A: A set where elements have degrees of membership between 0 and 1.

# Q2: How is a fuzzy set different from a crisp set?
# A: Crisp sets have 0/1 membership; fuzzy sets allow partial membership.

# Q3: What is the union operation in fuzzy sets?
# A: max(μ_A(x), μ_B(x))

# Q4: What is the intersection operation?
# A: min(μ_A(x), μ_B(x))

# Q5: How is the complement calculated?
# A: μ_A'(x) = 1 - μ_A(x)

# Q6: What are De Morgan’s Laws in fuzzy logic?
# A: (A ∪ B)' = A' ∩ B' and (A ∩ B)' = A' ∪ B'

# Q7: Do De Morgan’s Laws always hold exactly?
# A: No, they hold approximately due to decimal rounding.

# Q8: Why is fuzzy logic used in AI?
# A: For reasoning with uncertain/imprecise info in real-world scenarios.

# Q9: Give a real-life example of a fuzzy set.
# A: "Tall people" – someone 5'10" might have 0.6 membership in "tall".

# Q10: Applications of fuzzy logic?
# A: Washing machines, ACs, traffic lights, trading systems, etc.

# =============================================
#              END OF FUZZY LOGIC NOTES
# =============================================




########## nim ###########
# min-max (Nim Game) - 15
def min_max(stones, is_computer_turn):
    if stones == 0:
        return not is_computer_turn
    if is_computer_turn:
        for move in [1, 2, 3]:
            if stones - move >= 0 and min_max(stones - move, False):
                return True
            return False
    else:
        for move in [1, 2, 3]:
            if stones - move >= 0 and not min_max(stones - move, True):
                return False
            return True


def player_2(stones):
    for move in [1, 2, 3]:
        if stones - move >= 0 and min_max(stones - move, False):
            return move
        return 1


stones = int(input("Enter the no of stones : "))
turn = input("Who plays first ? ")

while stones > 0:
    print(f"Stones left : {stones}")
    if turn == "player2":
        move = player_2(stones)
        print(f"Player_2 removes {move} stones")
        stones -= move
        if stones == 0:
            print("Player 2 wins!")
            break
        turn = "user"
    else:
        move = int(input("Your move : (1-3)"))
        if move not in [1, 2, 3] or move > stones:
            print("Invalid move")
            continue
        stones -= move
        if stones == 0:
            print("You win")
            break
        turn = "player2"






########### MLP ###########
import numpy as np
import itertools
import random

# ------------------------------
# Common Helper Functions
# ------------------------------


# Activation functions
def binary_step(x):
    return 1 if x >= 0 else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Generate all binary input combinations
def generate_inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))


# Get user-defined targets
def get_targets(x, output_neurons=1):
    y = []
    print("\nEnter output values (0/1) for each input:")
    for xi in x:
        if output_neurons == 1:
            try:
                out = int(input(f"{list(xi)} → "))
                y.append([out if out in (0, 1) else 0])
            except:
                y.append([0])
        else:
            row = []
            for o in range(output_neurons):
                try:
                    out = int(input(f"{list(xi)} → Output {o+1}: "))
                    row.append(out if out in (0, 1) else 0)
                except:
                    row.append(0)
            y.append(row)
    return np.array(y)


# ------------------------------
# 1. Random Value MLP - 2 Hidden Layers - 1 Output (Binary Step)
# ------------------------------
def random_mlp_two_hidden_one_output():
    print("\n--- Random MLP: Two Hidden Layers, One Output ---\n")
    n = int(input("Enter number of binary inputs: "))
    x_data = generate_inputs(n)
    y_data = get_targets(x_data)

    tries = 0
    solved = False

    while not solved:
        tries += 1

        w1 = np.random.uniform(-2, 2, (n, 2))
        w2 = np.random.uniform(-2, 2, (2, 2))
        w3 = np.random.uniform(-2, 2, (2, 1))
        b1, b2, b3 = random.randint(-3, 3), random.randint(-3, 3), random.randint(-3, 3)

        # Forward pass
        h1 = np.vectorize(binary_step)(np.dot(x_data, w1) + b1)
        h2 = np.vectorize(binary_step)(np.dot(h1, w2) + b2)
        out = np.vectorize(binary_step)(np.dot(h2, w3) + b3)

        if np.array_equal(out, y_data):
            solved = True
            print(f"\n✅ Match found after {tries} tries:")
            print("Bias 1:", b1)
            print("Bias 2:", b2)
            print("Bias 3:", b3)
            print("W1:\n", w1)
            print("W2:\n", w2)
            print("W3:\n", w3)
            for i, y, o in zip(x_data, y_data, out):
                print(f"{list(i)} → Expected: {y[0]}, Output: {int(o[0])}")
        if tries % 100000 == 0:
            print(f"Attempt {tries}... still searching.")


# ------------------------------
# 2. Random Value MLP - 1 Hidden Layer - 2 Outputs (Binary Step)
# ------------------------------
def random_mlp_one_hidden_two_outputs():
    print("\n--- Random MLP: One Hidden Layer, Two Outputs ---\n")
    n = 4
    x_data = generate_inputs(n)
    y_data = get_targets(x_data, output_neurons=2)

    tries = 0
    solved = False

    while not solved:
        tries += 1

        w1 = np.random.uniform(-2, 2, (n, 3))
        w2 = np.random.uniform(-2, 2, (3, 2))
        b1, b2 = random.randint(-3, 3), random.randint(-3, 3)

        # Forward pass
        h1 = np.vectorize(binary_step)(np.dot(x_data, w1) + b1)
        out = np.vectorize(binary_step)(np.dot(h1, w2) + b2)

        if np.array_equal(out, y_data):
            solved = True
            print(f"\n✅ Match found after {tries} tries:")
            print("Bias 1:", b1)
            print("Bias 2:", b2)
            print("W1:\n", w1)
            print("W2:\n", w2)
            for i, y, o in zip(x_data, y_data, out):
                print(f"{list(i)} → Expected: {list(y)}, Output: {list(o)}")
        if tries % 100000 == 0:
            print(f"Attempt {tries}... still searching.")


# ------------------------------
# 3, 4, 5. MLP Class with Backpropagation
# ------------------------------
class SimpleMLP:
    def __init__(self, input_size, h1_size, h2_size, activation, lr=0.1):
        self.lr = lr
        self.activation = activation

        self.w1 = np.random.randn(input_size, h1_size)
        self.b1 = np.zeros((1, h1_size))

        self.w2 = np.random.randn(h1_size, h2_size)
        self.b2 = np.zeros((1, h2_size))

        self.w3 = np.random.randn(h2_size, 1)
        self.b3 = np.zeros((1, 1))

        # Set activation functions
        if activation == "sigmoid":
            self.act = sigmoid
            self.act_derivative = sigmoid_derivative
        elif activation == "relu":
            self.act = relu
            self.act_derivative = relu_derivative
        elif activation == "tanh":
            self.act = tanh
            self.act_derivative = tanh_derivative

    def forward(self, x):
        self.z1 = self.act(np.dot(x, self.w1) + self.b1)
        self.z2 = self.act(np.dot(self.z1, self.w2) + self.b2)
        self.output = self.act(np.dot(self.z2, self.w3) + self.b3)
        return self.output

    def backward(self, x, y):
        error = y - self.output
        d_output = error * self.act_derivative(self.output)

        error2 = d_output.dot(self.w3.T)
        d_z2 = error2 * self.act_derivative(self.z2)

        error1 = d_z2.dot(self.w2.T)
        d_z1 = error1 * self.act_derivative(self.z1)

        # Update weights and biases
        self.w3 += self.z2.T.dot(d_output) * self.lr
        self.b3 += np.sum(d_output, axis=0, keepdims=True) * self.lr

        self.w2 += self.z1.T.dot(d_z2) * self.lr
        self.b2 += np.sum(d_z2, axis=0, keepdims=True) * self.lr

        self.w1 += x.T.dot(d_z1) * self.lr
        self.b1 += np.sum(d_z1, axis=0, keepdims=True) * self.lr

    def train(self, x, y, epochs=10000):
        for i in range(epochs):
            self.forward(x)
            self.backward(x, y)
            if i % 1000 == 0:
                loss = np.mean((y - self.output) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, x):
        return (self.forward(x) > 0.5).astype(int)


def backprop_mlp(activation_type):
    print(
        f"\n--- Backpropagation MLP: Activation = {activation_type.capitalize()} ---\n"
    )
    n = int(input("Enter number of binary inputs: "))
    lr = float(input("Enter learning rate: "))
    hidden1 = int(input("Enter neurons in Hidden Layer 1: "))
    hidden2 = int(input("Enter neurons in Hidden Layer 2: "))

    x_data = generate_inputs(n)
    y_data = get_targets(x_data)

    model = SimpleMLP(n, hidden1, hidden2, activation=activation_type, lr=lr)
    model.train(x_data, y_data)

    print("\nResults after training:")
    preds = model.predict(x_data)
    for i, y, o in zip(x_data, y_data, preds):
        print(f"{list(i)} → Expected: {y[0]}, Predicted: {o[0]}")


# ------------------------------
# Menu to Run
# ------------------------------
if __name__ == "__main__":
    while True:
        print("\nSelect Assignment:")
        print("1. Random MLP (2 Hidden Layers, 1 Output)")
        print("2. Random MLP (1 Hidden Layer, 2 Outputs)")
        print("3. Backpropagation MLP (Sigmoid)")
        print("4. Backpropagation MLP (ReLU)")
        print("5. Backpropagation MLP (Tanh)")
        print("6. Exit")

        choice = input("\nEnter choice: ")

        if choice == "1":
            random_mlp_two_hidden_one_output()
        elif choice == "2":
            random_mlp_one_hidden_two_outputs()
        elif choice == "3":
            backprop_mlp("sigmoid")
        elif choice == "4":
            backprop_mlp("relu")
        elif choice == "5":
            backprop_mlp("tanh")
        elif choice == "6":
            break
        else:
            print("Invalid choice. Try again!")


# --------------------------------------------------------------
# 🎯 Multi-Layer Perceptron (MLP) - Quick Reference Cheat Sheet
# --------------------------------------------------------------

# 📚 What is MLP?
# - A type of neural network with one or more hidden layers.
# - Each layer has neurons, each neuron has weights and biases.
# - Neurons apply activation functions to produce output.

# 🔥 Basic Steps:
# 1. Forward Propagation: Inputs → Hidden Layers → Output Layer
# 2. Activation Functions applied at each neuron.
# 3. Compare Output with Target → Calculate Error.
# 4. Backward Propagation: Update Weights and Biases (only in learning models).

# 🧠 Important Concepts:
# - Inputs: Binary inputs (0 or 1)
# - Weights: Strength of connection between neurons.
# - Bias: Adjustment term to shift activation.
# - Activation Functions:
#     - Binary Step: 0 or 1 output.
#     - Sigmoid: Smooth output between 0 and 1.
#     - ReLU: Output is input if >0, else 0.
#     - Tanh: Output between -1 and +1.
# - Epoch: One full pass over the entire dataset.

# 🚀 Differences in Assignments:

# 1. Random MLP (2 Hidden Layers, 1 Output):
# - Randomly assign weights and biases.
# - No learning (no backpropagation).
# - Keep trying random values until the desired output is achieved.

# 2. Random MLP (1 Hidden Layer, 2 Outputs):
# - Similar to (1), but with two output neurons.
# - Still random trial, no learning.

# 3. Backpropagation MLP with Sigmoid Activation:
# - Proper learning using backpropagation.
# - Activation function is Sigmoid.
# - Weights and biases are updated based on error gradients.

# 4. Backpropagation MLP with ReLU Activation:
# - Learning using backpropagation.
# - Activation function is ReLU.
# - Faster convergence in many cases compared to Sigmoid.

# 5. Backpropagation MLP with Tanh Activation:
# - Learning using backpropagation.
# - Activation function is Tanh.
# - Output centered between -1 and +1 for better balance.

# ✨ Summary:
# - Random Models = Random Trial and Error (no real learning).
# - Backpropagation Models = Systematic Learning (reduces error over time).
# - Activation Functions decide the neuron behavior and impact learning speed.
# - Weights and Biases control how the MLP "learns" from inputs.

# 🌟 Bonus Tip:
# - Sigmoid: Good for simple binary outputs.
# - ReLU: Good for deeper and faster models.
# - Tanh: Good when needing zero-centered outputs.

# --------------------------------------------------------------
# END OF CHEAT SHEET
# --------------------------------------------------------------





########## NLP ###########
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from textblob import Word
import pandas as pd

# Download necessary NLTK data files (only once)
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


# Function to clean the text
def clean_text(text):
    # Remove punctuation, numbers, and special characters using regular expression
    text = re.sub(r"[^A-Za-z\s]", "", text)

    # Remove extra white spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Function to convert text to lowercase
def convert_to_lowercase(text):
    return text.lower()


# Function to perform tokenization
def tokenize_text(text):
    # Tokenize the text into words
    return word_tokenize(text)


# Function to remove stop words
def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


# Function to correct misspelled words
def correct_spelling(tokens):
    corrected_tokens = [Word(word).correct() for word in tokens]
    return corrected_tokens


# Function to perform stemming and lemmatization
def stem_and_lemmatize(text):
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Stemming and Lemmatization
    stemmed_words = [
        stemmer.stem(word) for word in tokens if word not in stopwords.words("english")
    ]
    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stopwords.words("english")
    ]

    return stemmed_words, lemmatized_words


# Function to generate 3 consecutive words (trigrams) after lemmatization
def generate_trigrams(lemmatized_words):
    # Generate trigrams (3 consecutive words)
    trigrams = list(ngrams(lemmatized_words, 3))
    return trigrams


# Function to perform One-Hot Encoding
def one_hot_encode(text_list):
    vectorizer = OneHotEncoder(sparse_output=False)
    one_hot_matrix = vectorizer.fit_transform([[text] for text in text_list])
    return one_hot_matrix


# Function to perform Bag of Words (BOW)
def bag_of_words(text_list):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(text_list)
    return bow_matrix.toarray(), vectorizer.get_feature_names_out()


# Function to perform TF-IDF
def tfidf(text_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_list)
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()


# Read the text from the sample files
with open("text_files/tech1.txt", "r") as file1, open(
    "text_files/tech2.txt", "r"
) as file2, open("text_files/tech3.txt", "r") as file3:
    text1 = file1.read()
    text2 = file2.read()
    text3 = file3.read()
with open("text_files/text_file.txt", "r") as file4:
    text_file = file4.read()


# Combine all three text files
combined_text = text1 + "\n" + text2 + "\n" + text3

# 1. Clean the text
cleaned_text = clean_text(text_file)

# 2. Convert the text to lowercase
lowercase_text = convert_to_lowercase(cleaned_text)

# 3. Perform Tokenization
tokens = tokenize_text(lowercase_text)

# 4. Remove stop words
tokens_no_stopwords = remove_stopwords(tokens)

# 5. Correct misspelled words
corrected_tokens = correct_spelling(tokens_no_stopwords)

# 6. Perform stemming and lemmatization
stemmed_words, lemmatized_words = stem_and_lemmatize(" ".join(corrected_tokens))

# 7. Create a list of 3 consecutive words after lemmatization (trigrams)
trigrams = generate_trigrams(lemmatized_words)

# 8. Perform One-Hot Encoding
one_hot_matrix = one_hot_encode(combined_text.split())
print("\nOne-Hot Encoding:")
print(one_hot_matrix)

# 9. Perform Bag of Words (BoW)
bow_matrix, bow_features = bag_of_words([combined_text])
print("\nBag of Words (BoW):")
print("Feature Names:", bow_features)
print("BoW Matrix:\n", bow_matrix)

# 10. Perform TF-IDF
tfidf_matrix, tfidf_features = tfidf([combined_text])
print("\nTF-IDF:")
print("Feature Names:", tfidf_features)
print("TF-IDF Matrix:\n", tfidf_matrix)

# Output the results
print("\nOriginal Text:\n", combined_text)
print("\nCleaned Text:\n", cleaned_text)
print("\nLowercase Text:\n", lowercase_text)
print("\nTokens:\n", tokens)
print("\nTokens without Stopwords:\n", tokens_no_stopwords)
print("\nCorrected Tokens:\n", corrected_tokens)
print("\nStemmed Words:\n", stemmed_words)
print("\nLemmatized Words:\n", lemmatized_words)
print("\nTrigrams:\n", trigrams)


"""
    ### **Assignment 22: Text Cleaning, Tokenization, Stop Words Removal, Misspelling Correction**
---

#### **Viva Questions and Answers:**
1. **What is the purpose of text cleaning in this assignment?**
   - **Answer:** It removes unwanted characters like punctuation, numbers, and extra spaces to make the text uniform for further processing.

2. **What is tokenization?**
   - **Answer:** Tokenization is the process of splitting a text into smaller units like words or phrases (tokens).

3. **What is the role of stop word removal?**
   - **Answer:** Stop words are common words like "the," "is," etc., that don't add significant meaning to the text. Removing them reduces noise in analysis.

4. **What is spell correction?**
   - **Answer:** Spell correction uses algorithms or libraries (like `TextBlob`) to correct common spelling mistakes in the text.

5. **What does `word_tokenize` do in tokenization?**
   - **Answer:** It splits the text into individual words using punctuation and whitespace as delimiters.

6. **Why do we use stemming and lemmatization in NLP?**
   - **Answer:** They reduce words to their root form to ensure different variations of a word (like "running" and "ran") are treated as the same word.

#### **Real-Life Application:**
- **Example:** In customer feedback analysis, cleaning text and removing stop words helps in accurately analyzing sentiments by focusing on important terms.
- **Application:** Used in sentiment analysis, spam filtering, and information retrieval systems.

---

### **Assignment 23: Stemming, Lemmatization, and Trigrams**
---

#### **Viva Questions and Answers:**
1. **What is the difference between stemming and lemmatization?**
   - **Answer:** Stemming reduces words to their base form (e.g., "running" → "run"), while lemmatization reduces words to their root word (e.g., "better" → "good").

2. **What is the purpose of generating trigrams?**
   - **Answer:** Trigrams help in understanding the context by analyzing sequences of three consecutive words in the text, which is useful for tasks like predictive text or language modeling.

3. **Why do we use `PorterStemmer`?**
   - **Answer:** `PorterStemmer` is a widely used algorithm for stemming words in NLP to reduce variations of words to a common root.

4. **How do lemmatization and stemming help in text analysis?**
   - **Answer:** Both help in reducing word variations, improving the consistency of text for better analysis and prediction.

#### **Real-Life Application:**
- **Example:** In automated text classification (e.g., spam vs. non-spam), stemming and lemmatization ensure that different forms of a word are treated as the same.
- **Application:** Used in search engines, chatbots, and language modeling for predictive typing.

---

### **Assignment 24: One-Hot Encoding on Technical Texts**
---

#### **Viva Questions and Answers:**
1. **What is One-Hot Encoding?**
   - **Answer:** One-Hot Encoding represents each word in the text as a vector with all zeros except for the position corresponding to that word, which is marked as 1.

2. **Why do we use One-Hot Encoding?**
   - **Answer:** It helps convert categorical data (words) into numerical form, allowing machine learning models to understand and process the data.

3. **What does `OneHotEncoder` do in scikit-learn?**
   - **Answer:** It transforms categorical values (like words) into a format that can be used for machine learning models by encoding each unique category as a binary vector.

#### **Real-Life Application:**
- **Example:** One-Hot Encoding is used in NLP applications like text classification to convert words into vectors for easier model processing.
- **Application:** Used in machine learning models, speech recognition, and recommendation systems.

---

### **Assignment 25: Bag of Words on Movie Reviews**
---

#### **Viva Questions and Answers:**
1. **What is the Bag of Words (BoW) model?**
   - **Answer:** BoW represents text as a set of words without considering grammar or word order but keeping track of word frequencies.

2. **Why is BoW important for text analysis?**
   - **Answer:** It simplifies text representation, making it easy for machine learning algorithms to process.

3. **What does `CountVectorizer` do?**
   - **Answer:** It converts text into a matrix of word counts, which can then be used for analysis or as input for machine learning models.

#### **Real-Life Application:**
- **Example:** In sentiment analysis, the Bag of Words model is used to classify reviews as positive or negative by counting word frequencies.
- **Application:** Used in text classification, sentiment analysis, and spam detection.

---

### **Assignment 26: TF-IDF on Tourist Reviews**
---

#### **Viva Questions and Answers:**
1. **What is TF-IDF?**
   - **Answer:** TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a corpus.

2. **Why is TF-IDF used in NLP?**
   - **Answer:** It helps identify important words by giving higher weights to words that are frequent in a specific document but rare in the entire corpus.

3. **What is the difference between TF-IDF and BoW?**
   - **Answer:** BoW counts word frequency, while TF-IDF adjusts word importance by considering the frequency of a word in a specific document relative to the entire corpus.

4. **How does `TfidfVectorizer` work?**
   - **Answer:** It transforms the text into a matrix where each word is weighted by its TF-IDF score, helping prioritize important words in text analysis.

#### **Real-Life Application:**
- **Example:** In a search engine, TF-IDF helps rank results by ensuring more relevant documents are prioritized based on unique keyword importance.
- **Application:** Used in information retrieval, document clustering, and search engine optimization.

---

### **General Application Question for All Assignments:**
- **How do these NLP techniques (Tokenization, Stop Words Removal, BoW, TF-IDF) apply to real-life systems?**
   - **Answer:** These techniques help in understanding and processing natural language for applications like sentiment analysis, machine translation, recommendation systems, and search engines. They allow systems to effectively understand, classify, and extract valuable insights from large volumes of text data.
"""