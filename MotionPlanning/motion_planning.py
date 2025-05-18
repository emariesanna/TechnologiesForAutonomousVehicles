import random
import osmnx as ox
import heapq
from math import sqrt
import time
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm

PLACE = "Rome"  # Options: Cagliari, Amalfi, Rome
ALGORITHM = "Astar"  # Options: Dijkstra, Astar
HEURISTIC = 1  # Options: 0 (euclidean), 1 (manhattan), 2 (chebyshev)

def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "#d36206"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def heuristicDistance(node, dest):
    dx = G.nodes[node]['x'] - G.nodes[dest]['x']
    dy = G.nodes[node]['y'] - G.nodes[dest]['y']

    if HEURISTIC == 0:
        return sqrt(dx**2 + dy**2)
    elif HEURISTIC == 1:
        return abs(dx) + abs(dy)
    elif HEURISTIC == 2:
        return max(abs(dx), abs(dy))
    else:
        raise ValueError("Invalid heuristic mode. Choose 0, 1, or 2.")

def a_star(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)

    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    open_set = [(heuristicDistance(orig, dest), orig)]
    g_score = {node: float("inf") for node in G.nodes}
    g_score[orig] = 0
    f_score = {node: float("inf") for node in G.nodes}
    f_score[orig] = heuristicDistance(orig, dest)

    came_from = {}
    step = 0

    while open_set:
        step += 1
        _, current = heapq.heappop(open_set)

        if current == dest:
            print("A* Iterations:", step)
            curr = dest
            while curr in came_from:
                G.nodes[curr]["previous"] = came_from[curr]
                curr = came_from[curr]
            return step

        G.nodes[current]["visited"] = True
        # print(f"Current: {current}")

        for u, v, key, data in G.out_edges(current, keys=True, data=True):
            neighbor = v
            weight = data.get("weight", float("inf"))
            tentative_g = g_score[current] + weight

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristicDistance(neighbor, dest)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                style_visited_edge((u, v, key))
                for _, n, k in G.out_edges(neighbor, keys=True):
                    style_active_edge((neighbor, n, k))
            
def dijkstra(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf") # distanza complessiva tra il nodo di partenza e il nodo considerato 
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)] # pqueue: lista ordinata crescente con i nodi da visitare
    step = 0
    while pq:
        _, node = heapq.heappop(pq) # estrae il primo nodo di pq

        if node == dest:
            print("Iterations:", step)
            #plot_graph()
            return step
        if G.nodes[node]["visited"]: continue # break
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node): # per ogni ramo del nodo visitato 
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1] # estraggo il nodo vicino
            weight = G.edges[(edge[0], edge[1], 0)]["weight"] # estraggo il peso del ramo
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight: # se la distanza del vicino è maggiore della distanza del nodo visitato + peso del ramo
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node # salvi nel vicino il nodo migliore per raggiungerlo
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor)) # aggiorniamo in pqueue il nuovo valore della distanza
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1

def plot_graph():
    ox.plot_graph(
        G,
        node_size=[G.nodes[node]["size"] for node in G.nodes],
        edge_color=[G.edges[edge]["color"] for edge in G.edges],
        edge_alpha=[G.edges[edge]["alpha"] for edge in G.edges],
        edge_linewidth=[G.edges[edge]["linewidth"] for edge in G.edges],
        node_color="white",
        bgcolor="#18080e"
    )

def reconstruct_path(orig, dest, plot=False, algorithm=None):
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    speeds = []
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000

def plot_heatmap(algorithm):
    edge_colors = ox.plot.get_edge_colors_by_attr(G, f"{algorithm}_uses", cmap="hot")
    fig, _ = ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        # edge_color = edge_colors,
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        bgcolor = "#18080e"
    )

def load_graph():
    global G
    if PLACE == "Cagliari":
        place_name = "Cagliari, Sardinia, Italy"
        G = ox.graph_from_place(place_name, network_type="drive")
    elif PLACE == "Amalfi":
        place_name = "Amalfi, Campania, Italy"
        G = ox.graph_from_place(place_name, network_type="drive")
    elif PLACE == "Rome":
        place_name = "Rome, Latium, Italy"
        G = ox.graph_from_place(place_name, network_type="drive")

def get_start_end_nodes():
    start = random.choice(list(G.nodes))
    end = random.choice(list(G.nodes))
    return start, end

def initialize_edges():
    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        maxspeed = 40
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]
            if type(maxspeed) == list:
                speeds = [ int(speed) for speed in maxspeed ]
                maxspeed = min(speeds)
            elif type(maxspeed) == str:
                maxspeed = maxspeed.strip(" mph")
                maxspeed = int(maxspeed)
        G.edges[edge]["maxspeed"] = maxspeed
        # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed

if __name__ == "__main__":
    print("Loading graph...")
    load_graph()
    print("Graph loaded")
    dijkstra_iterations_list = []
    dijkstra_times = []
    astar_iterations_list = []
    astar_times = []
    heuristicMode = 1 #Variabile globale

    for i in tqdm(range(100), desc="Simulazioni"):
        start, end = get_start_end_nodes()
        initialize_edges()
        # Dijkstra
        for edge in G.edges:
            G.edges[edge]["dijkstra_uses"] = 0
        # print("Running Dijkstra")
        start_time = time.time()
        dijkstra_iterations = dijkstra(start, end)
        dijkstra_time = time.time() - start_time
        dijkstra_iterations_list.append(dijkstra_iterations)
        dijkstra_times.append(dijkstra_time)
        # print(f"Done (Dijkstra time: {dijkstra_time:.4f} seconds)")
        reconstruct_path(start, end, algorithm="dijkstra", plot=True)
        #plot_heatmap("dijkstra")

        # A*
        for edge in G.edges:
            G.edges[edge]["astar_uses"] = 0
        # print("Running A*")
        start_time = time.time()
        astar_iterations = a_star(start, end)
        astar_time = time.time() - start_time
        astar_iterations_list.append(astar_iterations)
        astar_times.append(astar_time)
        # print(f"Done (A* time: {astar_time:.4f} seconds)")
        reconstruct_path(start, end, algorithm="astar", plot=True)
        #plot_heatmap("dijkstra")
        

    #plot_heatmap("astar")
    # Calcolo statistiche
    print("\nDijkstra Iterations: min={}, max={}, mean={:.2f}".format(
        min(dijkstra_iterations_list), max(dijkstra_iterations_list), sum(dijkstra_iterations_list)/len(dijkstra_iterations_list)))
    print("Dijkstra Times: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        min(dijkstra_times), max(dijkstra_times), sum(dijkstra_times)/len(dijkstra_times)))
    print("A* Iterations: min={}, max={}, mean={:.2f}".format(
        min(astar_iterations_list), max(astar_iterations_list), sum(astar_iterations_list)/len(astar_iterations_list)))
    print("A* Times: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        min(astar_times), max(astar_times), sum(astar_times)/len(astar_times)))
    
    x = list(range(1, len(dijkstra_iterations_list) + 1))
    # Iterazioni
    plt.figure(figsize=(12, 6))
    plt.plot(x, dijkstra_iterations_list, label="Dijkstra", linewidth=2)
    plt.plot(x, astar_iterations_list, label="A*", linewidth=2)
    plt.title("Andamento del numero di iterazioni per simulazione")
    plt.xlabel("Numero simulazione")
    plt.ylabel("Numero di iterazioni")
    plt.legend()
    plt.grid(True)
    plt.show()