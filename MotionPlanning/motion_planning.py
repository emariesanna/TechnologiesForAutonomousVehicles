import random
import osmnx as ox
import networkx as nx
import heapq
from math import sqrt
import time
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm

PLACE = "Rome"  # Options: Cagliari, Amalfi, Rome, Turin

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
    G.edges[edge]["linewidth"] = 2

def heuristicDistance(node, dest):
    # distanze in metri, velocità in km/h
    dx = G.nodes[node]['x'] - G.nodes[dest]['x']
    dy = G.nodes[node]['y'] - G.nodes[dest]['y']

    return sqrt(dx**2 + dy**2) / max_speed_limit

def a_star(orig, dest):
    # Inizializza i nodi e gli archi
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        # f_distance: distanza totale tra il nodo considerato e il nodo di arrivo (misurata + euristica)
        G.nodes[node]["f_distance"] = float("inf")
        # g_distance: componente misurata della distanza tra il nodo considerato e il nodo di arrivo
        G.nodes[node]["g_distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["g_distance"] = 0
    G.nodes[orig]["f_distance"] = heuristicDistance(orig, dest)
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    # open_set: lista ordinata crescente con i nodi da visitare
    open_set = [(heuristicDistance(orig, dest), orig)]
    step = 0

    while open_set:
        _, current = heapq.heappop(open_set) # estrae il primo nodo di open_set

        if current == dest: # se il nodo estratto è il nodo di destinazione, termina l'algoritmo
            return step
        
        # questa semplificazione è possibile perché l'euristica è consistente
        if G.nodes[current]["visited"]: continue # se il nodo corrente è già stato visitato, salta l'iterazione
        G.nodes[current]["visited"] = True

        # per ogni ramo del nodo corrente
        for i_node, f_node, key, data in G.out_edges(current, keys=True, data=True):
            # estrae il nodo vicino
            neighbor = f_node
            # estrae il peso del ramo
            weight = data["weight"]
            # calcola la nuova componente misurata per il vicino come 
            # somma tra quella del nodo corrente e il peso dle ramo
            tentative_g = G.nodes[current]["g_distance"] + weight

            # se la nuova componente misurata è minore della attuale componente misurata del vicino
            if tentative_g < G.nodes[neighbor]["g_distance"]:
                # salva nel vicino il nodo corrente, che è il nuovo nodo migliore per raggiungerlo
                G.nodes[neighbor]["previous"] = current
                # aggiorna componente misurata per il vicino al nuovo valore migliore
                G.nodes[neighbor]["g_distance"] = tentative_g
                # aggiorna la distanza totale per il vicino
                G.nodes[neighbor]["f_distance"] = tentative_g + heuristicDistance(neighbor, dest)
                # inserisce in open_set il vicino con il suo nuovo valore di distanza
                heapq.heappush(open_set, (G.nodes[neighbor]["f_distance"], neighbor))
                style_visited_edge((i_node, f_node, key))
                for _, n, k in G.out_edges(neighbor, keys=True):
                    style_active_edge((neighbor, n, k))
        step += 1
    raise ValueError("A*: No path found")            

def dijkstra(orig, dest):
    # Inizializza i nodi e gli archi
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        # distanza complessiva tra il nodo di partenza e il nodo considerato 
        G.nodes[node]["distance"] = float("inf") 
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
        _, current = heapq.heappop(pq) # estrae il primo nodo di pq

        if current == dest: # se il nodo corrente è il nodo di destinazione, termina l'algoritmo
            return step
        
        if G.nodes[current]["visited"]: continue # se il nodo corrente è già stato visitato, salta l'iterazione
        G.nodes[current]["visited"] = True

        for i_node, f_node, key, data in G.out_edges(current, keys=True, data=True): # per ogni ramo del nodo visitato 
            style_visited_edge((i_node, f_node, key))
            neighbor = f_node # estrae il nodo vicino
            weight = data["weight"] # estrae il peso del ramo
            # se la distanza del vicino è maggiore della distanza del nodo corrente + peso del ramo
            if G.nodes[neighbor]["distance"] > G.nodes[current]["distance"] + weight:
                # aggiorna la distanza del vicino al nuovo valore migliore, ovvero la distanza del nodo corrente + peso del ramo
                G.nodes[neighbor]["distance"] = G.nodes[current]["distance"] + weight
                # salva nel vicino il nodo corrente, che è il nuovo nodo migliore per raggiungerlo
                G.nodes[neighbor]["previous"] = current 
                # inserisce in pqueue il vicino con il suo nuovo valore di distanza
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor)) 
                for i_node2, f_node2 in G.out_edges(neighbor):
                    style_active_edge((i_node2, f_node2, 0))
        step += 1
    raise ValueError("Dijkstra: No path found")

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

def plot_heatmap(algorithm):
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = ox.plot.get_edge_colors_by_attr(G, f"{algorithm}_uses", cmap="hot"),
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        bgcolor = "#18080e"
    )

def reconstruct_path(orig, dest, algorithm=None):
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    speeds = []
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        key = list(G[prev][curr].keys())[0]
        dist += G.edges[(prev, curr, 0)]["length"]
        speeds.append(G.edges[(prev, curr, 0)]["maxspeed"])
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, key)][f"{algorithm}_uses"] = G.edges[(prev, curr, key)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000

def load_graph():
    global G
    if PLACE == "Cagliari":
        place_name = "Cagliari, Sardinia, Italy"
        G = ox.graph_from_place(place_name, network_type="drive")
        G =ox.project_graph(G)
    elif PLACE == "Amalfi":
        place_name = "Amalfi, Campania, Italy"
        G = ox.graph_from_place(place_name, network_type="drive")
        G =ox.project_graph(G)
    elif PLACE == "Rome":
        place_name = "Rome, Latium, Italy"
        G = ox.graph_from_place(place_name, network_type="drive")
        G =ox.project_graph(G)
    elif PLACE == "Turin":
        place_name = "Turin, Piedmont, Italy"
        G = ox.graph_from_place(place_name, network_type="drive")
        G =ox.project_graph(G)

def get_start_end_nodes():
    start = random.choice(list(G.nodes))
    end = random.choice(list(G.nodes))
    return start, end

def initialize_edges():
    # distanze in metri, velocità in km/h
    count_list = 0
    count_string = 0
    count_none = 0
    global max_speed_limit
    max_speed_limit = 0
    length_list = []
    unique_speed_limits = set()

    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        standard_speed_limit = 50
        if "maxspeed" in G.edges[edge]:
            speed_limit = G.edges[edge]["maxspeed"]
            if type(speed_limit) == list:
                count_list += 1
                speed_limits = [ int(speed) for speed in speed_limit ]
                speed_limit = min(speed_limits)
            elif type(speed_limit) == str:
                count_string += 1
                # speed_limit = speed_limit.strip(" mph")
                speed_limit = int(speed_limit)
        else:
            count_none += 1
            speed_limit = standard_speed_limit
        if speed_limit > max_speed_limit:
            max_speed_limit = speed_limit
        if speed_limit not in unique_speed_limits:
            unique_speed_limits.add(speed_limit)
        G.edges[edge]["maxspeed"] = speed_limit
        # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / speed_limit
        length_list.append(G.edges[edge]["length"])
    
    print(f"Unique speed limits: {sorted(unique_speed_limits)}")
    print(f"Maximum speed limit: {max_speed_limit} km/h")
    print(f"Minimum length: {min(length_list):.2f} m")
    print(f"Maximum length: {max(length_list):.2f} m")
    print(f"Average length: {sum(length_list)/len(length_list):.2f} m")
    if count_list:
        print(f"{count_list} edges have a list of speed limits")
    if count_string:
        print(f"{count_string} edges have a string speed limit")
    if count_none:
        print(f"{count_none} edges have no speed limit, using default value")

if __name__ == "__main__":
    print("Loading graph...")
    load_graph()
    print("Graph loaded")
    initialize_edges()
    dijkstra_iterations_list = []
    dijkstra_times = []
    astar_iterations_list = []
    astar_times = []

    for i in tqdm(range(100), desc="Simulazioni"):
        start, end = get_start_end_nodes()
        if not nx.has_path(G, start, end):
            print(f"\nNessun percorso tra {start} e {end}")
            continue
        # Dijkstra
        for edge in G.edges:
            G.edges[edge]["dijkstra_uses"] = 0
        # print("Running Dijkstra")
        start_time = time.time()
        dijkstra_iterations = dijkstra(start, end)
        dijkstra_time = time.time() - start_time
        dijkstra_iterations_list.append(dijkstra_iterations)
        dijkstra_times.append(dijkstra_time)
        # plot_graph()
        # print(f"Done (Dijkstra time: {dijkstra_time:.4f} seconds)")
        # reconstruct_path(start, end, algorithm="dijkstra")
        # plot_heatmap("dijkstra")

        # A*
        for edge in G.edges:
            G.edges[edge]["astar_uses"] = 0
        # print("Running A*")
        start_time = time.time()
        astar_iterations = a_star(start, end)
        astar_time = time.time() - start_time
        astar_iterations_list.append(astar_iterations)
        astar_times.append(astar_time)
        # plot_graph()
        # print(f"Done (A* time: {astar_time:.4f} seconds)")
        # reconstruct_path(start, end, algorithm="astar")
        # plot_heatmap("astar")
        
    print("\nDijkstra Iterations: min={}, max={}, mean={:.2f}".format(
        min(dijkstra_iterations_list), max(dijkstra_iterations_list), sum(dijkstra_iterations_list)/len(dijkstra_iterations_list)))
    print("Dijkstra Times: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        min(dijkstra_times), max(dijkstra_times), sum(dijkstra_times)/len(dijkstra_times)))
    print("A* Iterations: min={}, max={}, mean={:.2f}".format(
        min(astar_iterations_list), max(astar_iterations_list), sum(astar_iterations_list)/len(astar_iterations_list)))
    print("A* Times: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        min(astar_times), max(astar_times), sum(astar_times)/len(astar_times)))
    
    x = list(range(1, len(dijkstra_iterations_list) + 1))
    # Ordina entrambe le liste di iterazioni e tempi nello stesso ordine
    combined = sorted(zip(dijkstra_iterations_list, astar_iterations_list, dijkstra_times, astar_times))
    dijkstra_sorted, astar_sorted, dijkstra_times_sorted, astar_times_sorted = zip(*combined)
    dijkstra_sorted = list(dijkstra_sorted)
    astar_sorted = list(astar_sorted)
    dijkstra_times_sorted = list(dijkstra_times_sorted)
    astar_times_sorted = list(astar_times_sorted)

    # Iterazioni e tempi
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_iter_dijkstra = 'tab:blue'
    color_iter_astar = 'tab:cyan'
    color_time_dijkstra = 'tab:red'
    color_time_astar = 'tab:orange'
    ax1.set_xlabel("Numero simulazione")
    ax1.set_ylabel("Numero di iterazioni", color=color_iter_dijkstra)
    ax1.plot(x, dijkstra_sorted, label="Dijkstra Iterations", color=color_iter_dijkstra, linewidth=3)
    ax1.plot(x, astar_sorted, label="A* Iterations", color=color_iter_astar, linewidth=3)
    ax1.tick_params(axis='y', labelcolor=color_iter_dijkstra)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Tempo di esecuzione (s)", color=color_time_dijkstra)
    ax2.plot(x, dijkstra_times_sorted, label="Dijkstra Time", color=color_time_dijkstra, linewidth=2)
    ax2.plot(x, astar_times_sorted, label="A* Time", color=color_time_astar, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_time_dijkstra)
    ax2.legend(loc='upper right')
    plt.title("Numero di iterazioni e tempo di esecuzione per simulazione")
    plt.show()