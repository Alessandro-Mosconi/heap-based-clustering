import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from caseconverter import snakecase
from matplotlib.cm import get_cmap
from matplotlib.patches import Wedge
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from tabulate import tabulate
import matplotlib.pyplot as plt


def generate_route_matrix_numpy(nodes):
    """
    Calcola una matrice delle distanze euclidee tra nodi usando NumPy.
    Restituisce una matrice NumPy di shape (N, N).
    """
    coords = np.array(list(nodes.values()))  # N x 2 array
    dist_matrix = cdist(coords, coords)      # N x N matrice delle distanze
    print(f"✅ Matrice calcolata. Dimensione: {dist_matrix.shape}")
    return dist_matrix

def create_hosts(nodes):
    return [{"name": f"node-{node_id}", "labels": {}} for node_id in nodes]

def communities_to_cluster_data(communities, node_coords):
    # Crea mappatura node_id -> index (0-based)
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(sorted(node_coords.keys(), key=lambda x: int(x)))}

    # Estrai gli indici dalla mappa
    return [{
        "cluster": [
            node_id_to_index[member["name"].split("-")[1]]
            for member in community["members"]
        ]
    } for community in communities]



def plot_compactness_comparison(metrics_slpa, metrics_overlap,
                                            metrics_resource_expo, metrics_resource_lognorm):
    methods = ["SLPA", "Overlapping", "Resource-aware (expo)", "Resource-aware (lognorm)"]
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

    metrics = ["compactness_pairwise", "compactness_centroid", "compactness_radius", "density"]
    x = np.arange(len(metrics))
    width = 0.2

    # Prepara i valori
    values_by_method = {
        "SLPA": [metrics_slpa[m] for m in metrics],
        "Overlapping": [metrics_overlap[m] for m in metrics],
        "Resource-aware (expo)": [metrics_resource_expo[m] for m in metrics],
        "Resource-aware (lognorm)": [metrics_resource_lognorm[m] for m in metrics],
    }

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Per ogni metodo, plottiamo 4 barre (su 4 metriche), ma assegnandole agli assi giusti
    for i, (method, color) in enumerate(zip(methods, colors)):
        compactness_vals = values_by_method[method][:3]
        density_val = values_by_method[method][3]

        # Asse sinistro (compactness)
        ax1.bar(x[:3] + (i - 1.5) * width, compactness_vals, width, label=method if i == 0 else "", color=color)

        # Asse destro (density)
        ax2.bar(x[3] + (i - 1.5) * width, density_val, width, color=color)

    # Etichette e stile
    ax1.set_ylabel("Compactness Score")
    ax2.set_ylabel("Density")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=15)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    fig.suptitle("Compactness and Density Comparison (Dual Y-Axis)", fontsize=14)
    ax1.legend(methods, loc="upper left")

    plt.tight_layout()
    plt.savefig("compactness_and_density_split_axis.png", dpi=300, bbox_inches='tight')
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_detailed_computation_times(time_slpa, time_data_init_slpa,
                                    time_overlap, time_data_init_overlap,
                                    time_resource_expo, time_data_init_resource_expo,
                                    time_resource_lognorm, time_data_init_resource_lognorm,
                                    log_scale=False):

    methods = ['SLPA', 'Overlapping', 'Resource-aware (expo)', 'Resource-aware (lognorm)']

    computation_times = [time_slpa, time_overlap, time_resource_expo, time_resource_lognorm]
    init_times = [time_data_init_slpa, time_data_init_overlap, time_data_init_resource_expo, time_data_init_resource_lognorm]
    total_times = [c + i for c, i in zip(computation_times, init_times)]

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x - width, computation_times, width, label='Computation Time', color='#4c72b0')
    bars2 = plt.bar(x, init_times, width, label='Data Init Time', color='#dd8452')
    bars3 = plt.bar(x + width, total_times, width, label='Total Time', color='#55a868')

    if log_scale:
        plt.yscale("log")

    plt.ylabel("Time (ms)")
    title = "Computation & Initialization Time per Method" + (" (log)" if log_scale else "")
    plt.title(title)
    plt.xticks(x, methods, rotation=15)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:  # Avoid log(0) positioning issues
                plt.text(bar.get_x() + bar.get_width() / 2, yval * (1.05 if not log_scale else 1.2),
                         f"{yval:.0f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    filename = "detailed_computation_time" + ("_log" if log_scale else "") + ".png"
    plt.savefig(filename, format="png", dpi=300)
    print(f"✅ Grafico salvato come: {filename}")
    plt.show()




def print_compactness_table(metrics_slpa, metrics_overlap,
                            metrics_resource_expo, metrics_resource_lognorm):
    headers = ["Metric", "SLPA", "Overlapping", "Resource-aware (expo)", "Resource-aware (lognorm)"]
    table = []

    for metric in metrics_slpa:
        row = [
            metric,
            f"{metrics_slpa[metric]:.4f}",
            f"{metrics_overlap[metric]:.4f}",
            f"{metrics_resource_expo[metric]:.4f}",
            f"{metrics_resource_lognorm[metric]:.4f}"
        ]
        table.append(row)

    print("\n📊 Compactness Comparison Table")
    print(tabulate(table, headers=headers, tablefmt="grid"))


def print_execution_time_table(time_slpa, time_data_init_slpa,
                                time_overlap, time_data_init_overlap,
                                time_resource_expo, time_data_init_resource_expo,
                                time_resource_lognorm, time_data_init_resource_lognorm):
    headers = [
        "Time Metric", "SLPA", "Overlapping",
        "Resource-aware (expo)", "Resource-aware (lognorm)"
    ]

    table = [
        ["computation_time (ms)", time_slpa, time_overlap,
         time_resource_expo, time_resource_lognorm],

        ["data_initialization_time (ms)", time_data_init_slpa, time_data_init_overlap,
         time_data_init_resource_expo, time_data_init_resource_lognorm],

        ["total_time (ms)",
         time_slpa + time_data_init_slpa,
         time_overlap + time_data_init_overlap,
         time_resource_expo + time_data_init_resource_expo,
         time_resource_lognorm + time_data_init_resource_lognorm]
    ]

    print("\n⏱️ Execution Time Comparison Table")
    print(tabulate(table, headers=headers, tablefmt="grid"))



def save_compactness_table_as_image(metrics_slpa, metrics_overlap,
                                       metrics_resource_expo, metrics_resource_lognorm,
                                       filename="compactness_comparison_v2.png"):
    data = {
        "SLPA": [f"{metrics_slpa[m]:.4f}" for m in metrics_slpa],
        "Overlapping": [f"{metrics_overlap[m]:.4f}" for m in metrics_overlap],
        "Resource-aware (expo)": [f"{metrics_resource_expo[m]:.4f}" for m in metrics_resource_expo],
        "Resource-aware (lognorm)": [f"{metrics_resource_lognorm[m]:.4f}" for m in metrics_resource_lognorm]
    }

    index = list(metrics_slpa.keys())
    df = pd.DataFrame(data, index=index)

    fig, ax = plt.subplots(figsize=(11, len(df)*0.6 + 1))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                     cellLoc='center', loc='center', colLoc='center', rowLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.4)

    plt.title("Compactness Metrics Comparison", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Tabella salvata come immagine: {filename}")

def save_execution_time_table_as_image(time_slpa, time_data_init_slpa,
                                          time_overlap, time_data_init_overlap,
                                          time_resource_expo, time_data_init_resource_expo,
                                          time_resource_lognorm, time_data_init_resource_lognorm,
                                          filename="execution_time_comparison_v2.png"):

    computation = [time_slpa, time_overlap, time_resource_expo, time_resource_lognorm]
    initialization = [time_data_init_slpa, time_data_init_overlap, time_data_init_resource_expo, time_data_init_resource_lognorm]
    total = [c + i for c, i in zip(computation, initialization)]

    df = pd.DataFrame({
        "SLPA": [computation[0], initialization[0], total[0]],
        "Overlapping": [computation[1], initialization[1], total[1]],
        "Resource-aware (expo)": [computation[2], initialization[2], total[2]],
        "Resource-aware (lognorm)": [computation[3], initialization[3], total[3]]
    }, index=["computation_time (ms)", "data_initialization_time (ms)", "total_time (ms)"])

    fig, ax = plt.subplots(figsize=(11, len(df)*0.6 + 1))
    ax.axis('off')
    table = ax.table(cellText=df.values.astype(str), colLabels=df.columns, rowLabels=df.index,
                     cellLoc='center', loc='center', colLoc='center', rowLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.4)

    plt.title("Execution Time Breakdown", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Tabella salvata come immagine: {filename}")



### --- METRICA DI COMPATTEZZA --- ###
def normalize_scores(metrics_dict):
    all_metrics = list(metrics_dict.values())
    min_val = min(all_metrics)
    max_val = max(all_metrics)
    range_val = max_val - min_val if max_val != min_val else 1
    return {k: (v - min_val) / range_val for k, v in metrics_dict.items()}

def evaluate_compactness_metrics(cluster_data, route_matrix, nodes, positive_scores=True):
    def mean_pairwise_distance(ids):
        if len(ids) <= 1: return 0.0
        total = 0.0
        count = 0
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                total += route_matrix[ids[i], ids[j]]
                count += 1
        return total / count if count else 0.0

    def mean_centroid_distance(ids):
        if len(ids) <= 1: return 0.0
        coords = np.array([nodes[i] for i in ids])
        centroid = coords.mean(axis=0)
        return np.mean([np.linalg.norm(c - centroid) for c in coords])

    def cluster_radius(ids):
        if len(ids) <= 1: return 0.0
        coords = np.array([nodes[i] for i in ids])
        centroid = coords.mean(axis=0)
        return max([np.linalg.norm(c - centroid) for c in coords])

    def density(ids):
        if len(ids) < 3: return 0.0
        coords = np.array([nodes[i] for i in ids])
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

    # Cartella del dataset rispetto allo script corrente
    current_dir = os.path.dirname(__file__)  # directory dello script corrente
    overlap_result_folder = os.path.join(current_dir, "real_data_overlapped", "overlapped_result.json")
    original_result_folder = os.path.join(current_dir, "real_data_original", "orignal_result.json")
    resource_result_exp_folder = os.path.join(current_dir, "real_data_resource", "resource_result_exp.json")
    resource_result_log_folder = os.path.join(current_dir, "real_data_resource", "resource_result_log.json")
    resource_exp_list_path = os.path.join(current_dir, "real_data_unclustered", "resource_exp_list.json")
    resource_log_list_path = os.path.join(current_dir, "real_data_unclustered", "resource_log_list.json")
    filtered_nodes_path = os.path.join(current_dir, "real_data_unclustered", "filtered_nodes.json")

    if os.path.exists(filtered_nodes_path):
        print("📂 Caricamento nodes da 'filtered_nodes.json'")
        with open(filtered_nodes_path) as f:
            raw_nodes = json.load(f)
            nodes = {i: v for i, v in enumerate(raw_nodes.values())}
            print("Recuoperati " + str(len(nodes)) + "nodi")

    if os.path.exists(overlap_result_folder):
        print("📂 Caricamento nodes da 'overlapped_result.json'")
        with open(overlap_result_folder) as f:
            json_overlap = json.load(f)

    if os.path.exists(original_result_folder):
        print("📂 Caricamento nodes da 'orignal_result.json'")
        with open(original_result_folder) as f:
            json_original = json.load(f)

    if os.path.exists(resource_result_exp_folder):
        print("📂 Caricamento nodes da 'resource_result_exp.json'")
        with open(resource_result_exp_folder) as f:
            json_resource_exp = json.load(f)

    if os.path.exists(resource_result_log_folder):
        print("📂 Caricamento nodes da 'resource_result_log.json'")
        with open(resource_result_log_folder) as f:
            json_resource_log = json.load(f)

    if os.path.exists(resource_exp_list_path):
        print("📂 Caricamento nodes da 'resource_exp_list.json'")
        with open(resource_exp_list_path) as f:
            resource_exp_list = json.load(f)

    if os.path.exists(resource_log_list_path):
        print("📂 Caricamento nodes da 'resource_log_list.json'")
        with open(resource_log_list_path) as f:
            resource_log_list = json.load(f)

    route_matrix = generate_route_matrix_numpy(nodes)


    cluster_data_slpa = communities_to_cluster_data(json_original["communities"], raw_nodes)
    cluster_data_overlap = json_overlap["cluster_data"]
    cluster_data_resource_log = json_resource_log["cluster_data"]
    cluster_data_resource_exp = json_resource_exp["cluster_data"]

    metrics_slpa = evaluate_compactness_metrics(cluster_data_slpa, route_matrix, nodes)
    metrics_overlap = evaluate_compactness_metrics(cluster_data_overlap, route_matrix, nodes)
    metrics_resource_exp = evaluate_compactness_metrics(cluster_data_resource_exp, route_matrix, nodes)
    metrics_resource_log = evaluate_compactness_metrics(cluster_data_resource_log, route_matrix, nodes)



    # estrai tempi di esecuzione
    time_slpa = json_original["computation_time"]
    time_data_init_slpa = json_original["data_initialization_time"]
    time_overlap = json_overlap["computation_time"]
    time_data_init_overlap = json_overlap["data_initialization_time"]
    time_resource_log = json_resource_log["computation_time"]
    time_data_init_resource_log = json_resource_log["data_initialization_time"]
    time_resource_exp = json_resource_exp["computation_time"]
    time_data_init_resource_exp = json_resource_exp["data_initialization_time"]

    # stampa tabella
    print_compactness_table(metrics_slpa, metrics_overlap, metrics_resource_exp, metrics_resource_log)

    # stampa tempi
    print_execution_time_table(time_slpa, time_data_init_slpa,
                                time_overlap, time_data_init_overlap,
                                time_resource_exp, time_data_init_resource_exp,
                                time_resource_log, time_data_init_resource_log)

    # plot tempi
    plot_detailed_computation_times(time_slpa, time_data_init_slpa,
                                time_overlap, time_data_init_overlap,
                                time_resource_exp, time_data_init_resource_exp,
                                time_resource_log, time_data_init_resource_log)

    plot_detailed_computation_times(time_slpa, time_data_init_slpa,
                                time_overlap, time_data_init_overlap,
                                time_resource_exp, time_data_init_resource_exp,
                                time_resource_log, time_data_init_resource_log, log_scale=True)

    # plot compactness
    plot_compactness_comparison(metrics_slpa, metrics_overlap, metrics_resource_exp, metrics_resource_log)

    save_compactness_table_as_image(metrics_slpa, metrics_overlap, metrics_resource_exp, metrics_resource_log,
                                   filename="compactness_table_real_data.png")

    save_execution_time_table_as_image(time_slpa, time_data_init_slpa,
                                time_overlap, time_data_init_overlap,
                                time_resource_exp, time_data_init_resource_exp,
                                time_resource_log, time_data_init_resource_log,
                                   filename="time_table_real_data.png")

if __name__ == "__main__":
    main()
