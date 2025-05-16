import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import heapq
import json
import re

np.random.seed(42)

def save_cluster_data_simple(output_path, cluster_data, nodes, resource_list):
    output_data = {
        "num_clusters": len(cluster_data),
        "clusters": []
    }

    # Per statistiche globali
    total_resources_list = []
    cluster_node_counts = []

    for cluster_idx, data in enumerate(cluster_data):
        cluster_nodes = data["cluster"]
        total_res = data["total_resources"]
        node_count = len(cluster_nodes)

        total_resources_list.append(total_res)
        cluster_node_counts.append(node_count)

        cluster_info = {
            "cluster_id": int(cluster_idx + 1),
            "total_resources": int(total_res),
            "num_nodes": int(node_count),
            "nodes": [
                {
                    "node_id": int(node_id),
                    "x": float(nodes[node_id + 1][0]),
                    "y": float(nodes[node_id + 1][1]),
                    "resources": int(resource_list[node_id])
                } for node_id in cluster_nodes
            ]
        }

        output_data["clusters"].append(cluster_info)

    # Individua indici dei cluster con max/min risorse
    max_idx = total_resources_list.index(max(total_resources_list))
    min_idx = total_resources_list.index(min(total_resources_list))

    output_data["cluster_stats"] = {
        "min_resources": {
            "value": int(total_resources_list[min_idx]),
            "num_nodes": int(cluster_node_counts[min_idx])
        },
        "max_resources": {
            "value": int(total_resources_list[max_idx]),
            "num_nodes": int(cluster_node_counts[max_idx])
        },
        "avg_resources": float(sum(total_resources_list) / len(total_resources_list))
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

def generate_nodes(rows, cols):
    return {node_number: (i, j) for node_number, (i, j) in enumerate(((i, j) for i in range(rows) for j in range(cols)), start=1)}

def create_distance_matrix(nodes):
    coords = np.array(list(nodes.values()))
    return cdist(coords, coords)

def generate_resource_list(nodes):
    return [np.random.randint(1, 101) for _ in range(len(nodes))]  

def to_snake_case(text):
    return re.sub(r'[^a-zA-Z0-9]+', '_', text).lower()

def balanced_clustering_by_nodes_and_resources(route_matrix, resource_list, max_number_nodes,T = 50, tolerance = 1.1 ):
    n = route_matrix.shape[0]
    num_clusters = max(2, int(np.ceil(n / max_number_nodes)))
    total_resources = sum(resource_list)
    
    def initialize_centroids(k):
        centroids = [np.random.randint(0, n)]
        for _ in range(1, k):
            distances = np.min(route_matrix[centroids], axis=0)
            next_centroid = np.argmax(distances)
            centroids.append(next_centroid)
        return centroids

    centroids = initialize_centroids(num_clusters)
    avg_resources = total_resources / num_clusters
    
    print(f"Risorse medie per cluster: {avg_resources}")
    
    # Costruzione delle heap (una per centroide)
    node_assigned = np.full(n, False)
    clusters = {i: [] for i in range(num_clusters)}
    resources_in_cluster = {i: 0 for i in range(num_clusters)}
    node_count_in_cluster = {i: 0 for i in range(num_clusters)}

    heaps = []
    for i, c in enumerate(centroids):
        heap = []
        for node in range(n):
            if node != c:
                heapq.heappush(heap, (route_matrix[c, node], node))
        heaps.append(heap)
        clusters[i].append(c)
        node_assigned[c] = True
        resources_in_cluster[i] += resource_list[c]
        node_count_in_cluster[i] += 1

    # Assegnazione iterativa dei nodi
    changes = True
    while changes:
        changes = False
        for i in range(num_clusters):
            while heaps[i]:
                _, node = heapq.heappop(heaps[i])
                if node_assigned[node]:
                    continue
                projected_resource = resources_in_cluster[i] + resource_list[node]
                projected_count = node_count_in_cluster[i] + 1
                if projected_count <= max_number_nodes and projected_resource <= avg_resources * 1.1:
                    clusters[i].append(node)
                    node_assigned[node] = True
                    resources_in_cluster[i] = projected_resource
                    node_count_in_cluster[i] = projected_count
                    changes = True
                    break  # passa al prossimo cluster

    # Se ci sono nodi rimasti non assegnati, li assegnamo greedy
    for node in range(n):
        if not node_assigned[node]:
            distances = [(route_matrix[node, centroids[i]], i) for i in range(num_clusters)]
            distances.sort()
            for _, i in distances:
                if node_count_in_cluster[i] < max_number_nodes:
                    clusters[i].append(node)
                    resources_in_cluster[i] += resource_list[node]
                    node_count_in_cluster[i] += 1
                    node_assigned[node] = True
                    break

    # Bilanciamento finale: T iterazioni
    for _ in range(T):
        for i in range(num_clusters):

            if len(clusters[i]) <= 1:
                continue  # Nulla da spostare

            centroid_i = centroids[i]

            # Trova il nodo più lontano dal centroide, escludendo il centroide stesso
            distances = [(route_matrix[centroid_i, node], node) for node in clusters[i] if node != centroid_i]
            if not distances:
                continue

            distances.sort(reverse=True)
            node_to_move = distances[0][1]

            # Trova i centroidi più vicini al nodo_to_move (escludendo quello attuale)
            candidate_clusters = []
            for j in range(num_clusters):
                if j == i:
                    continue
                if node_count_in_cluster[j] >= max_number_nodes * tolerance:
                    continue
                projected_res = resources_in_cluster[j] + resource_list[node_to_move]
                if resources_in_cluster[j] >= resources_in_cluster[i]:
                    continue

                dist = route_matrix[node_to_move, centroids[j]]
                candidate_clusters.append((dist, j))

            if candidate_clusters:
                # Ordina per distanza e prendi il cluster migliore
                candidate_clusters.sort()
                best_j = candidate_clusters[0][1]

                # Esegui lo spostamento
                clusters[i].remove(node_to_move)
                clusters[best_j].append(node_to_move)

                resources_in_cluster[i] -= resource_list[node_to_move]
                node_count_in_cluster[i] -= 1

                resources_in_cluster[best_j] += resource_list[node_to_move]
                node_count_in_cluster[best_j] += 1

                print(f"Moved node {node_to_move} from {i+1} to {best_j+1}")



    result = []
    for i in range(num_clusters):
        result.append({
            "cluster": clusters[i],
            "total_resources": resources_in_cluster[i],
            "num_nodes": node_count_in_cluster[i]
        })

    return result, centroids

def plot_clusters(cluster_data, centroids, nodes, resource_list, title):
    plt.figure(figsize=(20, 20))
    colors = plt.cm.get_cmap('tab10', len(cluster_data))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    for cluster_idx, data in enumerate(cluster_data):
        cluster_nodes = [nodes[node_id + 1] for node_id in data["cluster"]]
        resources = [resource_list[node_id] for node_id in data["cluster"]]
        x, y = zip(*cluster_nodes)
        marker = markers[cluster_idx % len(markers)]
        min_size = 50
        max_size = 300
        min_res = min(resource_list)
        max_res = max(resource_list)

        sizes = [min_size + (res - min_res) / (max_res - min_res) * (max_size - min_size) for res in resources]

        plt.scatter(x, y, s=sizes, c=[colors(cluster_idx)], marker=marker, label=f'Cluster {cluster_idx + 1}', alpha=0.6)

    centroid_coords = [nodes[centroid + 1] for centroid in centroids]
    x_centroids, y_centroids = zip(*centroid_coords)
    plt.scatter(x_centroids, y_centroids, c='black', marker='*', s=200, label='Centroidi')

    plt.title(title, fontsize=16)
    plt.xlabel("Coordinata X", fontsize=14)
    plt.ylabel("Coordinata Y", fontsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(to_snake_case(title) + '.png', format="png", dpi=300)
    plt.show()

config = {
    "rows": 51,
    "cols": 51,
    "max_number_nodes": 200,
}

nodes = generate_nodes(config["rows"], config["cols"])
route_matrix = create_distance_matrix(nodes)
resource_list = generate_resource_list(nodes)

cluster_data, centroids = balanced_clustering_by_nodes_and_resources(route_matrix, resource_list, config["max_number_nodes"])

save_cluster_data_simple("resoconto.json", cluster_data, nodes, resource_list)

plot_clusters(cluster_data, centroids, nodes, resource_list, "Clustering bilanciato per risorse")

for idx, cluster in enumerate(cluster_data):
    print(f"Cluster {idx + 1}: Total Resources = {cluster['total_resources']}, Number of Nodes = {cluster['num_nodes']}")
