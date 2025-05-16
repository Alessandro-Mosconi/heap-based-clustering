import json
import os
import time

import pandas as pd
import requests
from caseconverter import snakecase
from matplotlib.cm import get_cmap
from matplotlib.patches import Wedge
from matplotlib.ticker import MaxNLocator
from scipy.spatial import ConvexHull
from tabulate import tabulate


def generate_nodes_from_local(folder, total_taxis=10357, log_every=500, target_date=None):
    """
    Estrae la prima coordinata GPS di ciascun taxi dal dataset locale.
    Se `target_date` è specificata, prende la prima coordinata registrata in quella data (formato: 'YYYY-MM-DD').
    """
    nodes = {}
    start_time = time.time()

    for taxi_id in range(1, total_taxis + 1):
        file_path = os.path.join(folder, f"{taxi_id}.txt")
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path, header=None, names=["taxi_id", "datetime", "longitude", "latitude"])
            df["datetime"] = pd.to_datetime(df["datetime"])

            if target_date:
                filtered = df[df["datetime"].dt.date == pd.to_datetime(target_date).date()]
                if filtered.empty:
                    continue
                first_row = filtered.iloc[0]
            else:
                first_row = df.iloc[0]

            nodes[taxi_id] = [first_row["latitude"], first_row["longitude"]]

        except Exception as e:
            print(f"[Taxi {taxi_id}] Errore: {e}")
            continue

        if taxi_id % log_every == 0:
            elapsed = time.time() - start_time
            print(f"[{taxi_id}/{total_taxis}] Taxi processati - tempo: {elapsed:.2f}s")

    print(f"✅ {len(nodes)} nodi generati.")
    return nodes

def filter_nodes_by_bounds(nodes, lat_min=39.6, lat_max=40.3, lon_min=115.75, lon_max=117.25):
    """
    Filtra i nodi mantenendo solo quelli entro i range specificati di latitudine e longitudine.
    """
    filtered = {
        taxi_id: coord
        for taxi_id, coord in nodes.items()
        if lat_min <= coord[0] <= lat_max and lon_min <= coord[1] <= lon_max
    }
    print(f"✅ Nodi filtrati: {len(filtered)} / {len(nodes)}")
    return filtered



def generate_route_matrix(nodes, log_every=500):
    taxi_ids = list(nodes.keys())
    coords = np.array([nodes[t] for t in taxi_ids])
    dist = cdist(coords, coords)

    route_matrix = {}
    total = len(taxi_ids)
    start_time = time.time()

    for i, t1 in enumerate(taxi_ids):
        route_matrix[str(t1)] = {str(t2): float(dist[i][j]) for j, t2 in enumerate(taxi_ids)}
        if (i + 1) % log_every == 0:
            elapsed = time.time() - start_time
            print(f"[{i + 1}/{total}] Riga matrice completata - tempo: {elapsed:.2f}s")

    print("✅ route_matrix generata.")
    return route_matrix

from scipy.spatial.distance import cdist


def generate_route_matrix_numpy(nodes):
    """
    Calcola una matrice delle distanze euclidee tra nodi usando NumPy.
    Restituisce una matrice NumPy di shape (N, N).
    """
    coords = np.array(list(nodes.values()))  # N x 2 array
    dist_matrix = cdist(coords, coords)      # N x N matrice delle distanze
    print(f"✅ Matrice calcolata. Dimensione: {dist_matrix.shape}")
    return dist_matrix


def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"💾 Salvato '{filename}'")

def generate_resource_list(nodes):
    return [np.random.randint(1, 101) for _ in range(len(nodes))]


def plot_nodes_with_resource_scores(
    nodes,
    scores,
    title,
    figsize=(12, 12),
    cmap='viridis',
    use_lognorm=False,
    apply_clipping=True,
    point_size=4,
    alpha=0.8
):
    """
    Plotta i nodi colorati in base allo score di risorse.

    - nodes: dict {id: [lat, lon]}
    - scores: list di score (senza node_id), deve avere lo stesso ordine di list(nodes.keys())
    """
    node_ids = list(nodes.keys())
    coords = [nodes[nid] for nid in node_ids]
    lats, lons = zip(*coords)
    scores = np.array(scores)

    # Clipping
    if apply_clipping:
        vmin, vmax = np.percentile(scores, 1), np.percentile(scores, 99)
    else:
        vmin, vmax = np.min(scores), np.max(scores)

    norm = LogNorm(vmin=vmin, vmax=vmax) if use_lognorm else Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        lons, lats,
        c=scores,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=alpha
    )
    plt.title(title)
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.colorbar(scatter, label="Score risorse")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.svg", format="svg")
    print(f"✅ Plot salvato in {title}.svg")
    plt.show()


from matplotlib.colors import Normalize, LogNorm
import numpy as np

def plot_nodes_with_resource_size(
    nodes,
    scores,
    title,
    figsize=(12, 12),
    min_size=1.0,
    max_size=100.0,
    use_lognorm=False,
    apply_clipping=True,
    color="dodgerblue",
    alpha=0.6,
    edge_color="black",
    edge_width=0.5
):
    """
    Plotta i nodi con dimensione proporzionale allo score di risorse.

    - nodes: dict {id: [lat, lon]}
    - scores: list di score (senza node_id), deve avere lo stesso ordine di list(nodes.keys())
    """
    node_ids = list(nodes.keys())
    coords = [nodes[nid] for nid in node_ids]
    lats, lons = zip(*coords)
    scores = np.array(scores)

    # Clipping
    if apply_clipping:
        vmin, vmax = np.percentile(scores, 1), np.percentile(scores, 99)
    else:
        vmin, vmax = np.min(scores), np.max(scores)

    # Normalizzazione
    if use_lognorm:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        norm_scores = norm(scores)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        norm_scores = norm(scores)

    sizes = min_size + (max_size - min_size) * norm_scores

    plt.figure(figsize=figsize)
    plt.scatter(
        lons, lats,
        s=sizes,
        color=color,
        alpha=alpha,
        edgecolors=edge_color,
        linewidths=edge_width
    )
    plt.title(title)
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.svg", format="svg")
    print(f"✅ Plot salvato in {title}.svg")
    plt.show()



def generate_exponential_scores(nodes, scale=10.0, seed=42):
    np.random.seed(seed)
    return np.random.exponential(scale=scale, size=len(nodes)).tolist()

def generate_lognormal_scores(nodes, mean=0.0, sigma=2.0, seed=42):
    np.random.seed(seed)
    return np.random.lognormal(mean=mean, sigma=sigma, size=len(nodes)).tolist()



def create_hosts(nodes):
    return [{"name": f"node-{node_id}", "labels": {}} for node_id in nodes]

def communities_to_cluster_data(communities):
    return [{
        "cluster": [int(member["name"].split("-")[1]) - 1 for member in community["members"]]
    } for community in communities]


### --- PLOT FUNCTIONS --- ###

import matplotlib.pyplot as plt


def plot_nodes(nodes, figsize=(10, 10), alpha=0.4, point_size=1.2):
    """
    Plotta i nodi usando le loro coordinate geografiche (latitudine, longitudine).

    Parametri:
    - nodes: dict {id: [lat, lon]}
    - figsize: dimensioni della figura
    - alpha: trasparenza dei punti
    - point_size: dimensione dei punti
    """
    coords = list(nodes.values())
    lats = [coord[0] for coord in coords]
    lons = [coord[1] for coord in coords]

    plt.figure(figsize=figsize)
    plt.scatter(lons, lats, s=point_size, alpha=alpha, marker='o')
    plt.title("Distribuzione dei nodi (taxi) - coordinate GPS")
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
    plt.savefig(snakecase(title) + '.png', format="png", dpi=300)
    # plt.show()

def plot_clusters_overlap(cluster_data, centroids, nodes, route_matrix, title):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.get_cmap('tab10', len(cluster_data))
    ax = plt.gca()

    for cluster_idx, data in enumerate(cluster_data):
        cluster_nodes = [nodes[node_id + 1] for node_id in data["cluster"]]
        x, y = zip(*cluster_nodes)
        plt.scatter(x, y, c=[colors(cluster_idx)], label=f'Cluster {cluster_idx + 1}', s=100)

        for node_id in data["shared_nodes"]:
            distances = [route_matrix[node_id, centroid] for centroid in centroids]
            closest_clusters = np.argsort(distances)[:2]
            shared_coord = nodes[node_id + 1]
            wedge1 = Wedge(shared_coord, 0.3, 0, 180, facecolor=colors(closest_clusters[0]), edgecolor='black')
            wedge2 = Wedge(shared_coord, 0.3, 180, 360, facecolor=colors(closest_clusters[1]), edgecolor='black')
            ax.add_patch(wedge1)
            ax.add_patch(wedge2)

    centroid_coords = [nodes[centroid + 1] for centroid in centroids]
    x_centroids, y_centroids = zip(*centroid_coords)
    plt.scatter(x_centroids, y_centroids, c='black', marker='*', s=200, label='Centroidi')

    plt.title(title, fontsize=16)
    plt.xlabel("Coordinata X", fontsize=14)
    plt.ylabel("Coordinata Y", fontsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.savefig(snakecase(title) + '.png', format="png", dpi=300)
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
    plt.savefig(snakecase(title) + '.png', format="png", dpi=300)
    # plt.show()

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
    plt.savefig("compactness_score.png", format="png", dpi=300)
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
    plt.savefig("computation_time.png", format="png", dpi=300)
    # plt.show()
    
def print_compactness_and_time_table(metrics_slpa, metrics_overlap, metrics_resource, time_slpa, time_overlap, time_resource):
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

def save_comparison_table_as_image(metrics_slpa, metrics_overlap, metrics_resource, time_slpa, time_overlap, time_resource, filename="compactness_comparison.png"):

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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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
        "rows": 150,
        "cols": 150,
        "min_nodes_per_cluster": 100,
        "min_shared_nodes": 50,
        "min_exclusive_nodes": 20,
    }
    # Cartella del dataset rispetto allo script corrente
    current_dir = os.path.dirname(__file__)  # directory dello script corrente
    dataset_folder = os.path.join(current_dir, "TAXI_DATASET", "taxi_log_2008_by_id")

    nodes_path = os.path.join(current_dir, "TAXI_DATASET", "nodes.json")
    filtered_nodes_path = os.path.join(current_dir, "TAXI_DATASET", "filtered_nodes.json")
    resource_exp_list_path = os.path.join(current_dir, "TAXI_DATASET", "resource_exp_list.json")
    resource_log_list_path = os.path.join(current_dir, "TAXI_DATASET", "resource_log_list.json")


    if os.path.exists(filtered_nodes_path):
        print("📂 Caricamento nodes da 'filtered_nodes.json'")
        with open(filtered_nodes_path) as f:
            nodes = json.load(f)
    else:
        # Se esiste lo carico, altrimenti lo genero e salvo
        if os.path.exists(nodes_path):
            print("📂 Caricamento nodes da 'nodes.json'")
            with open(nodes_path) as f:
                nodes = json.load(f)
        else:
            print("⚙️  Generazione nodi da file .txt...")
            nodes = generate_nodes_from_local(dataset_folder)
            save_to_json(nodes, nodes_path)

        nodes = filter_nodes_by_bounds(nodes)
        save_to_json(nodes, filtered_nodes_path)

    if os.path.exists(resource_exp_list_path):
        print("📂 Caricamento nodes da 'resource_exp_list.json'")
        with open(resource_exp_list_path) as f:
            resource_exp_list = json.load(f)
    else:
        resource_exp_list = generate_exponential_scores(nodes)
        save_to_json(resource_exp_list, resource_exp_list_path)

    if os.path.exists(resource_log_list_path):
        print("📂 Caricamento nodes da 'resource_log_list.json'")
        with open(resource_log_list_path) as f:
            resource_log_list = json.load(f)
    else:
        resource_log_list = generate_lognormal_scores(nodes)
        save_to_json(resource_log_list, resource_log_list_path)

    #plot_nodes(nodes)
    #plot_nodes_with_resource_size(nodes,resource_exp_list, 'resource_exp_list_size')
    #plot_nodes_with_resource_size(nodes,resource_log_list, 'resource_log_list_size', use_lognorm=True)

    #plot_nodes_with_resource_scores(nodes,resource_exp_list, 'resource_exp_list')
    #plot_nodes_with_resource_scores(nodes,resource_log_list, 'resource_log_list', use_lognorm=True)
    # Genera matrice delle distanze
    route_matrix = generate_route_matrix_numpy(nodes)


    input_request = {
        "hosts": create_hosts(nodes),
        "delay-matrix": { "routes": route_matrix.tolist() }
    }

    # --- Clustering resources ---
    print("Sending resource exp")
    input_request.update({
        "resources": resource_exp_list,
        "max_number_nodes": 780
    })
    json_resource = requests.post("http://localhost:4567/api/communities/resource", json=input_request).json()
    print("Saving resource exp result")
    json.dump(json_resource, open('resource_result_exp.json', "w"), indent=4)
    print("Resource exp saved")


    # --- Clustering bilanciato per risorse ---
    print("Sending resource log")
    input_request.update({
        "resources": resource_log_list,
        "max_number_nodes": 780
    })
    json_resource = requests.post("http://localhost:4567/api/communities/resource", json=input_request).json()
    print("Saving resource log result")
    json.dump(json_resource, open('resource_result_log.json', "w"), indent=4)
    print("Resource log saved")



    # --- Clustering con overlap ---
    print("Sending overlap")
    input_request["parameters"] = {
        "min_nodes_per_cluster": 750,
        "min_shared_nodes": 200,
        "min_exclusive_nodes": 200
    }
    json_overlap = requests.post("http://localhost:4567/api/communities/overlap", json=input_request).json()
    print("Saving overlap result")
    json.dump(json_overlap, open('overlapped_result.json', "w"), indent=4)
    print("Overlap saved")


    # --- Clustering standard ---
    print("Sending SLPA")
    input_request["parameters"] = {
        "community-size": 780,
        "maximum-delay": 3,
        "iterations": 20,
        "probability-threshold": 30
    }
    json_original = requests.post("http://localhost:4567/api/communities", json=input_request).json()
    print("Enhancing SLPA result")
    for community in json_original.get("communities", []):
        for member in community.get("members", []):
            node_id = int(member["name"].split("-")[1])
            if node_id in nodes:
                x, y = nodes[node_id]
                member["labels"]["x"] = x
                member["labels"]["y"] = y
    print("Saving SLPA result")
    json.dump(json_original, open('orignal_result.json', "w"), indent=4)
    print("SLPA saved")





if __name__ == "__main__":
    main()
