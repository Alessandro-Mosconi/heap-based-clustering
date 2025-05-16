from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.patches import Wedge
from scipy.spatial.distance import cdist
import requests
import json
from caseconverter import snakecase
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas as pd
import random
from tabulate import tabulate

out_dir = "50x50_random"

### --- GENERAZIONE E UTILITIES --- ###

def generate_nodes(rows, cols):
    return {
        node_number: (i, j)
        for node_number, (i, j) in enumerate(((i, j) for i in range(rows) for j in range(cols)), start=1)
    }

def generate_random_nodes(rows, cols, seed=42):
    random.seed(seed)
    # Griglia espansa
    extended_coords = [(i, j) for i in range(rows * 2) for j in range(cols * 2)]
    # Mescolo le coordinate
    random.shuffle(extended_coords)
    # Prendo solo il numero desiderato di nodi
    selected_coords = extended_coords[:rows * cols]
    # Assegno gli ID da 1 in poi
    return {
        node_number: coord
        for node_number, coord in enumerate(selected_coords, start=1)
    }

def create_distance_matrix(nodes):
    coords = np.array(list(nodes.values()))
    return cdist(coords, coords)

def generate_resource_list(nodes):
    return [np.random.randint(1, 101) for _ in range(len(nodes))]

def create_hosts(nodes):
    return [{"name": f"node-{node_id}", "labels": {}} for node_id in nodes]

def communities_to_cluster_data(communities):
    return [{
        "cluster": [int(member["name"].split("-")[1]) - 1 for member in community["members"]]
    } for community in communities]


### --- PLOT FUNCTIONS --- ###

def plot_communities(communities, nodes, title):
    plt.figure(figsize=(10, 10))
    colors = get_cmap("tab10", len(communities))
    ax = plt.gca()

    for idx, community in enumerate(communities):
        node_ids = [int(member["name"].split("-")[1]) for member in community["members"]]
        coords = [nodes[node_id] for node_id in node_ids]
        x, y = zip(*coords)
        plt.scatter(x, y, c=[colors(idx)], label=community["name"], s=100, edgecolors='black')

    # estetica coerente
    plt.title(title, fontsize=16)
    plt.xlabel("Coordinata X", fontsize=14)
    plt.ylabel("Coordinata Y", fontsize=14)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir + "/" + snakecase(title) + '.png', format="png", dpi=300)
    # plt.show()

def plot_clusters_overlap(cluster_data, centroids, nodes, route_matrix, title):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    colors = plt.cm.get_cmap('tab20', len(cluster_data))  # più colori disponibili
    scatter_size = 100
    wedge_radius = 0.8  # scegli tu il valore ottimale in base alla densità

    for cluster_idx, data in enumerate(cluster_data):
        # Nodi esclusivi (senza contorno nero)
        cluster_nodes = [nodes[node_id + 1] for node_id in data["cluster"]]
        x, y = zip(*cluster_nodes)
        plt.scatter(x, y, c=[colors(cluster_idx)], label=f'Cluster {cluster_idx + 1}', s=scatter_size)

        # Nodi condivisi (con doppio colore e contorno nero)
        for node_id in data["shared_nodes"]:
            distances = [route_matrix[node_id, centroid] for centroid in centroids]
            closest_clusters = np.argsort(distances)[:2]
            shared_coord = nodes[node_id + 1]

            wedge1 = Wedge(shared_coord, wedge_radius, 0, 180,
                           facecolor=colors(closest_clusters[0]),
                           edgecolor='black', linewidth=1.0)
            wedge2 = Wedge(shared_coord, wedge_radius, 180, 360,
                           facecolor=colors(closest_clusters[1]),
                           edgecolor='black', linewidth=1.0)
            ax.add_patch(wedge1)
            ax.add_patch(wedge2)

    # Centroidi
    centroid_coords = [nodes[centroid + 1] for centroid in centroids]
    x_centroids, y_centroids = zip(*centroid_coords)
    plt.scatter(x_centroids, y_centroids, c='black', marker='*', s=200, label='Centroidi')

    # Dettagli grafici
    plt.title(title, fontsize=16)
    plt.xlabel("Coordinata X", fontsize=14)
    plt.ylabel("Coordinata Y", fontsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir + "/" + snakecase(title) + '.png', format="png", dpi=300)
    # plt.show()

def plot_clusters_resource(cluster_data, centroids, nodes, resource_list, title):
    plt.figure(figsize=(20, 20))
    colors = plt.cm.get_cmap('tab10', len(cluster_data))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    min_res, max_res = min(resource_list), max(resource_list)
    min_size, max_size = 50, 300

    for cluster_idx, data in enumerate(cluster_data):
        cluster_nodes = [nodes[node_id + 1] for node_id in data["cluster"]]
        resources = [resource_list[node_id] for node_id in data["cluster"]]
        sizes = [min_size + (res - min_res) / (max_res - min_res) * (max_size - min_size) for res in resources]
        x, y = zip(*cluster_nodes)
        marker = markers[cluster_idx % len(markers)]
        plt.scatter(x, y, s=sizes, c=[colors(cluster_idx)], marker=marker, label=f'Cluster {cluster_idx + 1}', alpha=0.6)

    centroid_coords = [nodes[centroid + 1] for centroid in centroids]
    x_centroids, y_centroids = zip(*centroid_coords)
    plt.scatter(x_centroids, y_centroids, c='black', marker='*', s=200, label='Centroidi')

    plt.title(title, fontsize=16)
    plt.xlabel("Coordinata X", fontsize=14)
    plt.ylabel("Coordinata Y", fontsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.savefig(out_dir + "/" + snakecase(title) + '.png', format="png", dpi=300)
    # plt.show()

def plot_nodes_with_resource_size(
        nodes,
        scores,
        title,
        figsize=(10, 10),
        min_size=1.0,
        max_size=100.0,
        use_lognorm=False,
        apply_clipping=True,
        color="dodgerblue",
        alpha=0.6,
        edge_color="black",
        edge_width=0.5
):
    node_ids = list(nodes.keys())
    xs, ys = zip(*(nodes[nid] for nid in node_ids))
    scores = np.array(scores)

    if apply_clipping:
        vmin, vmax = np.percentile(scores, 1), np.percentile(scores, 99)
    else:
        vmin, vmax = scores.min(), scores.max()

    if use_lognorm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    norm_scores = norm(scores)
    sizes = min_size + (max_size - min_size) * norm_scores

    plt.figure(figsize=figsize)
    plt.scatter(xs, ys,
                s=sizes,
                color=color,
                alpha=alpha,
                edgecolors=edge_color,
                linewidths=edge_width)
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + "/" + "plot_unclustered_nodes_resources.png", format="png", dpi=300)
    print(f"✅ Plot salvato in {title}.svg")
    plt.show()


def plot_nodes(nodes, figsize=(10, 10), point_size=100, alpha=0.6,
                   edge_color="black", edge_width=0.5, color="dodgerblue"):
    xs, ys = zip(*nodes.values())

    plt.figure(figsize=figsize)
    plt.scatter(
        xs, ys,
        s=point_size,
        color=color,
        alpha=alpha,
        edgecolors=edge_color,
        linewidths=edge_width
    )

    # estetica
    plt.title("Node distribution", fontsize=16)
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Y Coordinate", fontsize=14)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(out_dir + "/" + "plot_unclustered_nodes.png", format="png", dpi=300)
    plt.show()

def plot_compactness_comparison(metrics_slpa, metrics_overlap, metrics_resource):
    labels = list(metrics_slpa.keys())
    x = np.arange(len(labels))
    width = 0.25

    slpa = [metrics_slpa[k] for k in labels]
    overlap = [metrics_overlap[k] for k in labels]
    resource = [metrics_resource[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, slpa, width, label='SLPA')
    ax.bar(x, overlap, width, label='Overlapping')
    ax.bar(x + width, resource, width, label='Resource-aware')

    ax.set_ylabel("Compactness Score")
    ax.set_title("Compactness Comparison by Clustering Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir + "/" + "compactness_score.png", format="png", dpi=300)
    # plt.show()

def plot_computation_times(time_slpa, time_overlap, time_resource):
    methods = ['SLPA', 'Overlapping', 'Resource-aware']
    times = [time_slpa, time_overlap, time_resource]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, times, color=['#4c72b0', '#dd8452', '#55a868'])
    plt.title("Computation Time per Method")
    plt.ylabel("Time (ms)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.0f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(out_dir + "/" + "computation_time.png", format="png", dpi=300)
    # plt.show()

def print_compactness_and_time_table(metrics_slpa, metrics_overlap, metrics_resource,
                                     time_slpa, time_overlap, time_resource):
    headers = ["Metric", "SLPA", "Overlapping", "Resource-aware"]
    table = []

    for metric in metrics_slpa:
        row = [
            metric,
            f"{metrics_slpa[metric]:.4f}",
            f"{metrics_overlap[metric]:.4f}",
            f"{metrics_resource[metric]:.4f}"
        ]
        table.append(row)

    # aggiungiamo il tempo di esecuzione alla fine
    table.append([
        "computation_time (ms)",
        str(time_slpa),
        str(time_overlap),
        str(time_resource)
    ])

    print("\n📊 Compactness + Time Comparison Table")
    print(tabulate(table, headers=headers, tablefmt="grid"))

def save_comparison_table_as_image(metrics_slpa, metrics_overlap, metrics_resource,
                                   time_slpa, time_overlap, time_resource,
                                   filename="compactness_comparison.png"):

    # Costruisci dataframe
    data = {
        "SLPA": [f"{metrics_slpa[m]:.4f}" for m in metrics_slpa] + [str(time_slpa)],
        "Overlapping": [f"{metrics_overlap[m]:.4f}" for m in metrics_overlap] + [str(time_overlap)],
        "Resource-aware": [f"{metrics_resource[m]:.4f}" for m in metrics_resource] + [str(time_resource)]
    }

    index = list(metrics_slpa.keys()) + ["computation_time (ms)"]
    df = pd.DataFrame(data, index=index)

    # Plot con matplotlib
    fig, ax = plt.subplots(figsize=(9, len(df)*0.6 + 1))  # altezza dinamica
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                     cellLoc='center', loc='center', colLoc='center', rowLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.4)

    plt.title("Compactness Metrics & Computation Time Comparison", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir + "/" + filename, dpi=300, bbox_inches='tight')
    print(f"✅ Tabella salvata come immagine: {filename}")


### --- METRICA DI COMPATTEZZA --- ###


def evaluate_compactness_metrics(cluster_data, route_matrix, nodes, positive_scores=True):
    def mean_pairwise_distance(ids):
        if len(ids) <= 1: return 0.0
        total = 0.0
        count = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                total += route_matrix[ids[i], ids[j]]
                count += 1
        return total / count if count else 0.0

    def mean_centroid_distance(ids):
        if len(ids) <= 1: return 0.0
        coords = np.array([nodes[i + 1] for i in ids])
        centroid = coords.mean(axis=0)
        return np.mean([np.linalg.norm(c - centroid) for c in coords])

    def cluster_radius(ids):
        if len(ids) <= 1: return 0.0
        coords = np.array([nodes[i + 1] for i in ids])
        centroid = coords.mean(axis=0)
        return max([np.linalg.norm(c - centroid) for c in coords])

    def density(ids):
        if len(ids) < 3: return 0.0
        coords = np.array([nodes[i + 1] for i in ids])
        try:
            area = ConvexHull(coords).volume
        except:
            return 0.0
        return len(ids) / (area + 1e-6)

    raw_scores = {
        "mean_pairwise": [],
        "mean_centroid": [],
        "radius": [],
        "density": []
    }

    for cluster in cluster_data:
        ids = cluster["cluster"]
        raw_scores["mean_pairwise"].append(mean_pairwise_distance(ids))
        raw_scores["mean_centroid"].append(mean_centroid_distance(ids))
        raw_scores["radius"].append(cluster_radius(ids))
        raw_scores["density"].append(density(ids))

    if not positive_scores:
        return {metric: np.mean(raw_scores[metric]) for metric in raw_scores}

    # Converti tutte in "più grande è meglio"
    converted_scores = {
        "compactness_pairwise": np.mean([1 / (x + 1e-6) for x in raw_scores["mean_pairwise"]]),
        "compactness_centroid": np.mean([1 / (x + 1e-6) for x in raw_scores["mean_centroid"]]),
        "compactness_radius": np.mean([1 / (x + 1e-6) for x in raw_scores["radius"]]),
        "density": np.mean(raw_scores["density"])
    }

    return converted_scores


### --- MAIN --- ###

def main():
    config = {
        "rows": 50,
        "cols": 50,
        "min_nodes_per_cluster": 100,
        "min_shared_nodes": 50,
        "min_exclusive_nodes": 20,
    }

    #nodes = generate_nodes(config["rows"], config["cols"])
    nodes = generate_random_nodes(config["rows"], config["cols"])
    route_matrix = create_distance_matrix(nodes)
    resource_list = generate_resource_list(nodes)

    plot_nodes(nodes)
    plot_nodes_with_resource_size(
        nodes, resource_list,
        title="Node distribution with resource evidence"
    )

    input_request = {
        "hosts": create_hosts(nodes),
        "delay-matrix": { "routes": route_matrix.tolist() }
    }

    # --- Clustering bilanciato per risorse ---
    print("mando resource")
    input_request.update({
        "resources": resource_list,
        "max_number_nodes": 120
    })
    json_resource = requests.post("http://localhost:4567/api/communities/3", json=input_request).json()
    json.dump(json_resource, open('50x50_random/resource_result.json', "w"), indent=4)
    plot_clusters_resource(json_resource['cluster_data'], json_resource['centroids'], nodes, resource_list, "Clustering bilanciato per risorse")


    # --- Clustering con overlap ---
    print("mando overlap")
    input_request["parameters"] = {
        "min_nodes_per_cluster": config["min_nodes_per_cluster"],
        "min_shared_nodes": config["min_shared_nodes"],
        "min_exclusive_nodes": config["min_exclusive_nodes"]
    }
    json_overlap = requests.post("http://localhost:4567/api/communities/2", json=input_request).json()
    json.dump(json_overlap, open('50x50_random/roverlapped_result.json', "w"), indent=4)
    plot_clusters_overlap(json_overlap['cluster_data'], json_overlap['centroids'], nodes, route_matrix, "Clustering con centroidi e nodi condivisi")


    # --- Clustering standard ---
    print("mando originale")
    input_request["parameters"] = {
        "community-size": 100,
        "maximum-delay": 20,
        "iterations": 20,
        "probability-threshold": 30
    }
    json_original = requests.post("http://localhost:4567/api/communities", json=input_request).json()
    for community in json_original.get("communities", []):
        for member in community.get("members", []):
            node_id = int(member["name"].split("-")[1])
            if node_id in nodes:
                x, y = nodes[node_id]
                member["labels"]["x"] = x
                member["labels"]["y"] = y
    json.dump(json_original, open('50x50_random/orignal_result.json', "w"), indent=4)
    plot_communities(json_original["communities"], nodes, "Clustering SLPA")

    cluster_data_slpa = communities_to_cluster_data(json_original["communities"])
    cluster_data_overlap = json_overlap["cluster_data"]
    cluster_data_resource = json_resource["cluster_data"]

    metrics_slpa = evaluate_compactness_metrics(cluster_data_slpa, route_matrix, nodes)
    metrics_overlap = evaluate_compactness_metrics(cluster_data_overlap, route_matrix, nodes)
    metrics_resource = evaluate_compactness_metrics(cluster_data_resource, route_matrix, nodes)

    # estrai tempi di esecuzione
    time_slpa = json_original["computation_time"]
    time_overlap = json_overlap["computation_time"]
    time_resource = json_resource["computation_time"]

    # stampa tabella + tempi
    print_compactness_and_time_table(metrics_slpa, metrics_overlap, metrics_resource,
                                    time_slpa, time_overlap, time_resource)

    # plot tempi
    plot_computation_times(time_slpa, time_overlap, time_resource)

    # plot compactness
    plot_compactness_comparison(metrics_slpa, metrics_overlap, metrics_resource)

    save_comparison_table_as_image(metrics_slpa, metrics_overlap, metrics_resource,
                                   time_slpa, time_overlap, time_resource,
                                   filename="50x50_random/compactness_table.png")




if __name__ == "__main__":
    main()
