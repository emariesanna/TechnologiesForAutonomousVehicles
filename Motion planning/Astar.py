import osmnx as ox
import random
import heapq

def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "#d36206"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    #G.edges[edge]["color"] = "#d36206"
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    #G.edges[edge]["color"] = '#e8a900'
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def plot_graph():
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "#18080e"
    )

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

        if step == 0:
            print("Coordinata X node: "+str(G.nodes[node]['x']))
        
        if node == dest:
            print("Iterations:", step)
            #plot_graph()
            return
        if G.nodes[node]["visited"]: continue # break
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node): # per ogni ramo del nodo visitato 
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1] # estraggo il nodo vicino
            weight = G.edges[(edge[0], edge[1], 0)]["weight"] # estraggo il peso del ramo
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight: # se 
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1

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
        #edge_color = edge_colors,
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        bgcolor = "#18080e"
    )

#place_name = "Piedmont, California, USA"
#place_name = "Rome, Latium, Italy"
place_name = "Cagliari, Sardinia, Italy"
#place_name = "Turin, Piedmont, Italy"
G = ox.graph_from_place(place_name, network_type="drive")

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


for edge in G.edges:
    G.edges[edge]["dijkstra_uses"] = 0

start = random.choice(list(G.nodes))
end = random.choice(list(G.nodes))

print("Running Dijkstra")
dijkstra(start, end)
print( "Done")

reconstruct_path(start, end, algorithm="dijkstra", plot=True)
plot_heatmap("dijkstra")

