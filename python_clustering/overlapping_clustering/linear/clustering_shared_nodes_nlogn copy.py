import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from scipy.spatial.distance import cdist
from heapq import heappush, heappop
import json
from caseconverter import snakecase

# Generazione dei nodi
def generate_nodes(rows, cols):
    return {node_number: (i, j) for node_number, (i, j) in enumerate(((i, j) for i in range(rows) for j in range(cols)), start=1)}

# Creazione della matrice delle distanze
def create_distance_matrix(nodes):
    coords = np.array(list(nodes.values()))
    return cdist(coords, coords)

# Popolamento bilanciato dei cluster
def populate_clusters(route_matrix, clusters, centroids, cluster_size, assigned):
    num_clusters = len(clusters)
    heaps = [[] for _ in range(num_clusters)]

    for i in range(num_clusters):
        for j in range(route_matrix.shape[0]):
            if not assigned[j]:
                heappush(heaps[i], (route_matrix[centroids[i], j], j))

    for _ in range(route_matrix.shape[0]):
        for i in range(num_clusters):
            if len(clusters[i]) < cluster_size:
                while heaps[i]:
                    _, nearest_node = heappop(heaps[i])
                    if not assigned[nearest_node]:
                        clusters[i].append(nearest_node)
                        assigned[nearest_node] = True
                        break

# Calcolo dei nodi condivisi
def add_shared_nodes(route_matrix, clusters, centroids, shared_nodes, min_shared_nodes, min_exclusive_nodes):
    num_clusters = len(clusters)
    n = route_matrix.shape[0]

    distances_to_centroids = np.array([
        [route_matrix[node_idx, centroid] for centroid in centroids]
        for node_idx in range(n)
    ])

    for i in range(num_clusters):
        while len(shared_nodes.get(i, [])) < min_shared_nodes:
            heap = []
            for cluster_idx, cluster in enumerate(clusters):
                if cluster_idx == i:
                    continue

                # Calcola il numero di nodi esclusivi rimanenti
                exclusive_nodes = [
                    node for node in cluster
                    if node not in [n for nodes in shared_nodes.values() for n in nodes]
                ]

                # Salta il cluster se non ha abbastanza nodi esclusivi
                if len(exclusive_nodes) <= min_exclusive_nodes:
                    continue

                for node in exclusive_nodes:
                    heappush(heap, (distances_to_centroids[node, i], node))

            if not heap:
                raise ValueError(f"Non ci sono abbastanza nodi per garantire {min_shared_nodes} nodi condivisi nel cluster {i + 1}.")

            _, nearest_node = heappop(heap)
            original_cluster = next(
                cluster_idx for cluster_idx, cluster in enumerate(clusters) if nearest_node in cluster
            )

            shared_nodes.setdefault(i, []).append(nearest_node)
            shared_nodes.setdefault(original_cluster, []).append(nearest_node)

            if nearest_node not in clusters[i]:
                clusters[i].append(nearest_node)

    return shared_nodes

# Clustering bilanciato
def balanced_clustering(route_matrix, min_nodes_per_cluster, min_shared_nodes, min_exclusive_nodes):
    n = route_matrix.shape[0]
    num_clusters = n // min_nodes_per_cluster
    cluster_size = n // num_clusters

    if min_shared_nodes + min_exclusive_nodes >= cluster_size:
        raise ValueError("La somma di min_shared_nodes e min_exclusive_nodes deve essere minore della dimensione del cluster.")

    clusters = [[] for _ in range(num_clusters)]
    assigned = np.zeros(n, dtype=bool)
    centroids = [0]
    cached_dists = np.full(route_matrix.shape[1], np.inf)

    while len(centroids) < num_clusters:
        new_dists = route_matrix[centroids[-1]]
        cached_dists = np.minimum(cached_dists, new_dists)
        centroids.append(np.argmax(cached_dists))

    for i, centroid in enumerate(centroids):
        clusters[i].append(centroid)
        assigned[centroid] = True

    populate_clusters(route_matrix, clusters, centroids, cluster_size, assigned)

    unassigned_indices = np.where(~assigned)[0]
    for node in unassigned_indices:
        distances_to_centroids = [route_matrix[node, centroid] for centroid in centroids]
        clusters[np.argmin(distances_to_centroids)].append(node)

    shared_nodes = {}
    shared_nodes = add_shared_nodes(route_matrix, clusters, centroids, shared_nodes, min_shared_nodes, min_exclusive_nodes)

    result = []
    for i, cluster in enumerate(clusters):
        exclusive_nodes = [node for node in cluster if node not in [n for nodes in shared_nodes.values() for n in nodes]]
        shared_nodes_list = shared_nodes.get(i, [])
        result.append({
            "cluster": cluster,
            "exclusive_nodes": exclusive_nodes,
            "shared_nodes": shared_nodes_list
        })

    return result, centroids

# Funzione per salvare i dati

def save_cluster_data(output_path, cluster_data, nodes):
    output_data = {}
    output_data['num_clusters'] = len(cluster_data)
    output_data['clusters'] = []
    for cluster_idx, data in enumerate(cluster_data):
        cluster_info = {
            "cluster_id": int(cluster_idx + 1),
            "num_exclusive_nodes": int(len(data["exclusive_nodes"])),
            "num_shared_nodes": int(len(data["shared_nodes"])),
            "exclusive_nodes": [
                {"node_id": int(node_id), "x": float(nodes[node_id + 1][0]), "y": float(nodes[node_id + 1][1])} for node_id in data["exclusive_nodes"]
            ],
            "shared_nodes": [
                {
                    "node_id": int(node_id),
                    "x": float(nodes[node_id + 1][0]),
                    "y": float(nodes[node_id + 1][1]),
                    "clusters": [
                        int(idx + 1) for idx, cluster in enumerate(cluster_data) if node_id in cluster["shared_nodes"]
                    ]
                } for node_id in data["shared_nodes"]
            ]
        }
        output_data['clusters'].append(cluster_info)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

# Funzione per il plot

def plot_clusters(cluster_data, centroids, nodes, route_matrix, title):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.get_cmap('tab10', len(cluster_data))

    for cluster_idx, data in enumerate(cluster_data):
        cluster_nodes = [nodes[node_id + 1] for node_id in data["cluster"]]
        x, y = zip(*cluster_nodes)
        plt.scatter(x, y, c=[colors(cluster_idx)], label=f'Cluster {cluster_idx + 1}', s=100)

    centroid_coords = [nodes[centroid + 1] for centroid in centroids]
    x_centroids, y_centroids = zip(*centroid_coords)
    plt.scatter(x_centroids, y_centroids, c='black', marker='*', s=200, label='Centroidi')

    ax = plt.gca()
    for cluster_idx, data in enumerate(cluster_data):
        shared_coords = [nodes[node_id + 1] for node_id in data["shared_nodes"]]
        for node_id in data["shared_nodes"]:
            distances = [route_matrix[node_id, centroid] for centroid in centroids]
            closest_clusters = np.argsort(distances)[:2]  # Prendi i due cluster più vicini
            color1 = colors(closest_clusters[0])
            color2 = colors(closest_clusters[1])
            shared_coord = nodes[node_id + 1]
            wedge1 = Wedge(shared_coord, 0.3, 0, 180, facecolor=color1, edgecolor='black')
            wedge2 = Wedge(shared_coord, 0.3, 180, 360, facecolor=color2, edgecolor='black')
            ax.add_patch(wedge1)
            ax.add_patch(wedge2)

    plt.title(title, fontsize=16)
    plt.xlabel("Coordinata X", fontsize=14)
    plt.ylabel("Coordinata Y", fontsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.savefig(snakecase(title) + '.png', format="png", dpi=300)
    plt.show()

# Configurazione
config = {
    "rows": 40,
    "cols": 40,
    "min_nodes_per_cluster": 180,
    "min_shared_nodes": 50,
    "min_exclusive_nodes": 20,
}

nodes = generate_nodes(config["rows"], config["cols"])
route_matrix = create_distance_matrix(nodes)
cluster_data, centroids = balanced_clustering(route_matrix, config["min_nodes_per_cluster"], config["min_shared_nodes"], config["min_exclusive_nodes"])

save_cluster_data("resoconto.json", cluster_data, nodes)
'''
copy_cluster_data = []
for i, data in enumerate(cluster_data):
    copy_cluster_data.append({
            "cluster": cluster_data[i]['cluster'],
            "exclusive_nodes": cluster_data[i]['exclusive_nodes'],
            "shared_nodes": []
        })
plot_clusters(copy_cluster_data, centroids, nodes, route_matrix, "Clustering con centroidi")

'''
plot_clusters(cluster_data, centroids, nodes, route_matrix, "Clustering con centroidi e nodi condivisi")
