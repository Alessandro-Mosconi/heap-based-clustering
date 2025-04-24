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

def communities_to_cluster_data(communities):
    return [{
        "cluster": [int(member["name"].split("-")[1]) - 1 for member in community["members"]]
    } for community in communities]


### --- PLOT FUNCTIONS --- ###
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
    plt.savefig('node_plot.svg', format="svg")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_nodes_with_resource_size(
    nodes,
    scores,
    title="nodes_plot",
    figsize=(15, 15),
    min_size=5.0,
    max_size=80.0,
    alpha=0.6,
    color="dodgerblue",
    edge_color="black",
    edge_width=0.4
):
    """
    Plotta i nodi con dimensione ∝ score, colore uniforme, senza cluster né centroidi.

    Parametri:
    - nodes: dict {id: [lat, lon]}
    - scores: lista di score corrispondente all'ordine di nodes
    - title: titolo del plot e nome del file SVG
    - min_size: dimensione minima del punto
    - max_size: dimensione massima del punto
    - alpha: trasparenza dei punti
    - color: colore uniforme dei punti
    - edge_color: colore del bordo
    - edge_width: spessore del bordo
    """
    node_ids = list(nodes.keys())
    scores = np.array(scores)

    # Normalizza le dimensioni in base allo score
    vmin, vmax = np.percentile(scores, 1), np.percentile(scores, 99)
    norm = (scores - vmin) / (vmax - vmin)
    sizes = min_size + norm * (max_size - min_size)

    # Coordinate
    coords = [nodes[nid] for nid in node_ids if nid in nodes]
    lats, lons = zip(*coords)

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
    """
    Plotta i cluster con nodi condivisi, centroidi e colori ben distinti (fino a 20).
    """
    plt.figure(figsize=(10, 10))
    cmap = plt.colormaps['tab20']  # ColorBrewer-style palette fino a 20 colori
    ax = plt.gca()

    for cluster_idx, data in enumerate(cluster_data):
        color = cmap(cluster_idx % 20)  # Ricicla i colori se > 20
        cluster_nodes = [nodes[node_id] for node_id in data["cluster"]]
        x, y = zip(*cluster_nodes)
        plt.scatter(x, y, c=[color], label=f'Cluster {cluster_idx + 1}', s=100)
        colors = [cmap(i % 20) for i in range(20)]
        for node_id in data["shared_nodes"]:
            dist_row = route_matrix[node_id]
            distances = dist_row[centroids]
            closest_clusters = np.argpartition(distances, 2)[:2]
            closest_clusters = closest_clusters[np.argsort(distances[closest_clusters])]

            shared_coord = nodes[node_id]
            wedge1 = Wedge(shared_coord, 0.3, 0, 180, facecolor=colors[closest_clusters[0] % 20], edgecolor='black')
            wedge2 = Wedge(shared_coord, 0.3, 180, 360, facecolor=colors[closest_clusters[1] % 20], edgecolor='black')
            ax.add_patch(wedge1)
            ax.add_patch(wedge2)

    centroid_coords = [nodes[centroid] for centroid in centroids]
    x_centroids, y_centroids = zip(*centroid_coords)
    plt.scatter(x_centroids, y_centroids, c='black', marker='*', s=200, label='Centroidi')

    plt.title(title, fontsize=16)
    plt.xlabel("Longitudine", fontsize=14)
    plt.ylabel("Latitudine", fontsize=14)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig(snakecase(title) + '.png', format="png", dpi=300)
    plt.show()
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from collections import defaultdict
import time
def plot_clusters_overlap_2(
    nodes,
    centroids=None,
    cluster_data=None,
    title="plot_clusters_overlap",
    figsize=(15, 15),
    alpha=0.4,
    point_size=1.5,
    wedge_scale=0.001
):
    """
    Plotta i nodi GPS colorati per cluster, con nodi condivisi in wedge e centroidi come stelle.

    Corretto per:
    - proporzioni coerenti (grafico quadrato)
    - griglia visibile
    - gestione originale delle dimensioni e logiche invariata
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps["tab20"]
    colors = [cmap(i % 20) for i in range(20)]

    shared_nodes_set = set()
    if cluster_data:
        for data in cluster_data:
            shared_nodes_set.update(data.get("shared_nodes", []))

    clustered_coords = {}
    plotted_nodes = set()

    if cluster_data:
        for cluster_idx, data in enumerate(cluster_data):
            coords = clustered_coords.setdefault(cluster_idx, [])
            for node_id in data["cluster"]:
                if node_id not in nodes or node_id in shared_nodes_set:
                    continue
                lat, lon = nodes[node_id]
                coords.append((lon, lat))
                plotted_nodes.add(node_id)

    for cluster_idx in clustered_coords:
        lons, lats = zip(*clustered_coords[cluster_idx]) if clustered_coords[cluster_idx] else ([], [])
        if lons:
            ax.scatter(
                lons,
                lats,
                color=colors[cluster_idx % 20],
                s=point_size,
                alpha=alpha,
                edgecolors='none',
                label=f"Cluster {cluster_idx + 1}"
            )

    shared_node_to_clusters = {}
    for cluster_idx, data in enumerate(cluster_data):
        for node_id in data.get("shared_nodes", []):
            shared_node_to_clusters.setdefault(node_id, []).append(cluster_idx)

    r = wedge_scale * (point_size ** 0.5)
    for node_id, cluster_ids in shared_node_to_clusters.items():
        if node_id not in nodes or len(cluster_ids) < 2:
            continue
        lat, lon = nodes[node_id]
        color1 = colors[cluster_ids[0] % 20]
        color2 = colors[cluster_ids[1] % 20]
        wedge1 = Wedge((lon, lat), r, 0, 180, facecolor=color1, edgecolor='black', linewidth=0.05)
        wedge2 = Wedge((lon, lat), r, 180, 360, facecolor=color2, edgecolor='black', linewidth=0.05)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)
        plotted_nodes.add(node_id)

    unclustered = [nid for nid in nodes if nid not in plotted_nodes]
    if unclustered:
        lon_unc, lat_unc = zip(*[nodes[n] for n in unclustered])
        ax.scatter(
            lon_unc,
            lat_unc,
            color="gray",
            s=point_size,
            alpha=alpha,
            edgecolors='none',
            label="Non assegnati"
        )

    if centroids:
        centroid_coords = [nodes[c] for c in centroids if c in nodes]
        lat_c, lon_c = zip(*centroid_coords)
        ax.scatter(lon_c, lat_c, c='black', marker='*', s=10)

    ax.set_title("Distribuzione dei nodi con cluster", fontsize=14)
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.grid(True)  # ✅ Mostra griglia
    ax.set_aspect('equal', adjustable='box')  # ✅ Grafico quadrato
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
    plt.tight_layout()
    plt.savefig(f"{title}.svg", format="svg")
    print(f"✅ Plot salvato in {title}.svg")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


def plot_clusters_overlap_2(
        nodes,
        centroids=None,
        cluster_data=None,
        title="plot_clusters_overlap",
        figsize=(12, 10),
        alpha=0.7,
        point_size=1.8,
        wedge_scale=0.0008
):
    """
    Plots GPS nodes colored by cluster.
    Shared nodes are displayed as bicolored dots (wedge).
    Centroids are indicated with a black star.

    Modified to stretch vertically and remove white space.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = plt.colormaps["tab20"]
    colors = [cmap(i % 20) for i in range(20)]

    shared_nodes_set = set()
    if cluster_data:
        for data in cluster_data:
            shared_nodes_set.update(data.get("shared_nodes", []))

    clustered_coords = {}
    plotted_nodes = set()

    if cluster_data:
        for cluster_idx, data in enumerate(cluster_data):
            coords = clustered_coords.setdefault(cluster_idx, [])
            for node_id in data["cluster"]:
                if node_id not in nodes or node_id in shared_nodes_set:
                    continue
                lat, lon = nodes[node_id]
                coords.append((lon, lat))
                plotted_nodes.add(node_id)

    for cluster_idx in clustered_coords:
        lons, lats = zip(*clustered_coords[cluster_idx]) if clustered_coords[cluster_idx] else ([], [])
        if lons:
            ax.scatter(
                lons,
                lats,
                color=colors[cluster_idx % 20],
                s=point_size,
                alpha=alpha,
                edgecolors='none',
                label=f"Cluster {cluster_idx + 1}"
            )

    shared_node_to_clusters = {}
    for cluster_idx, data in enumerate(cluster_data):
        for node_id in data.get("shared_nodes", []):
            shared_node_to_clusters.setdefault(node_id, []).append(cluster_idx)

    r = wedge_scale * (point_size ** 0.5)
    for node_id, cluster_ids in shared_node_to_clusters.items():
        if node_id not in nodes or len(cluster_ids) < 2:
            continue
        lat, lon = nodes[node_id]
        color1 = colors[cluster_ids[0] % 20]
        color2 = colors[cluster_ids[1] % 20]
        wedge1 = Wedge((lon, lat), r, 0, 180, facecolor=color1, edgecolor='black', linewidth=0.05)
        wedge2 = Wedge((lon, lat), r, 180, 360, facecolor=color2, edgecolor='black', linewidth=0.05)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)
        plotted_nodes.add(node_id)

    unclustered = [nid for nid in nodes if nid not in plotted_nodes]
    if unclustered:
        lon_unc, lat_unc = zip(*[nodes[n] for n in unclustered])
        ax.scatter(
            lon_unc,
            lat_unc,
            color="gray",
            s=point_size,
            alpha=alpha,
            edgecolors='none',
            label="Non assegnati"
        )

    if centroids:
        centroid_coords = [nodes[c] for c in centroids if c in nodes]
        lat_c, lon_c = zip(*centroid_coords)
        ax.scatter(lon_c, lat_c, c='black', marker='*', s=10)

    # Calcola i limiti dei dati
    all_lons = [lon for (lat, lon) in nodes.values()]
    all_lats = [lat for (lat, lon) in nodes.values()]

    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    # Imposta i limiti dell'asse x (longitudine)
    ax.set_xlim(lon_min, lon_max)

    # Imposta i limiti dell'asse y (latitudine) per occupare tutto lo spazio
    ax.set_ylim(lat_min, lat_max)

    # Disabilita 'equal aspect' per evitare spazio bianco
    ax.set_aspect('auto')  # Oppure commenta questa linea

    ax.set_title("Distribuzione dei nodi con cluster", fontsize=14)
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.grid(True)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3)

    plt.savefig(f"{title}.svg", format="svg", bbox_inches='tight')
    print(f"✅ Plot salvato in {title}.svg")
    plt.show()


import plotly.express as px
import plotly.graph_objects as go

def plot_clusters_html(nodes, cluster_data, centroids=None, filename="plot_clusters.html"):
    # Prepara i dati
    data = []
    colors = px.colors.qualitative.T10 + px.colors.qualitative.Dark24

    # Nodo → cluster
    node_to_cluster = {}
    shared_nodes = set()
    for cluster_idx, data_cluster in enumerate(cluster_data):
        for nid in data_cluster["cluster"]:
            node_to_cluster[nid] = cluster_idx
        shared_nodes.update(data_cluster.get("shared_nodes", []))

    # Righe per ogni nodo
    for nid, (lat, lon) in nodes.items():
        cluster = node_to_cluster.get(nid, None)
        color = colors[cluster % len(colors)] if cluster is not None else "gray"
        marker_symbol = "circle"
        size = 6
        border_color = None

        if nid in shared_nodes:
            marker_symbol = "circle-open"  # oppure "x", "diamond", "star"
            size = 8

        if nid in centroids:
            marker_symbol = "star"
            color = "black"
            size = 10

        data.append(dict(
            lon=lon,
            lat=lat,
            id=nid,
            cluster=str(cluster) if cluster is not None else "none",
            color=color,
            marker=marker_symbol,
            size=size
        ))

    # Plot
    fig = go.Figure()

    for c in set(d["cluster"] for d in data):
        cluster_data = [d for d in data if d["cluster"] == c]
        fig.add_trace(go.Scattergeo(
            lon=[d["lon"] for d in cluster_data],
            lat=[d["lat"] for d in cluster_data],
            mode='markers',
            marker=dict(
                symbol=[d["marker"] for d in cluster_data],
                size=[d["size"] for d in cluster_data],
                color=[d["color"] for d in cluster_data],
                line=dict(width=0.5, color='black')
            ),
            name=f"Cluster {c}" if c != "none" else "Non assegnati",
            text=[f"Nodo {d['id']}" for d in cluster_data],
        ))

    fig.update_layout(
        title="Distribuzione dei nodi (HTML interattivo)",
        geo=dict(
            showland=True,
            showcountries=True,
            lataxis_range=[min(d['lat'] for d in data), max(d['lat'] for d in data)],
            lonaxis_range=[min(d['lon'] for d in data), max(d['lon'] for d in data)],
        )
    )

    fig.write_html(filename)
    print(f"✅ Plot HTML salvato in: {filename}")

def plot_communities_2(nodes, communities_json, figsize=(15, 15), alpha=0.4, point_size=1.5, save_svg=True):
    """
    Plotta le comunità usando i colori distinti, senza nodi condivisi o centroidi.

    Parametri:
    - nodes: dict {id: [lat, lon]}
    - communities_json: dizionario con struttura {"communities": [{"members": [{"name": "node-123"}]}]}
    - figsize: dimensione del grafico
    - alpha: trasparenza dei nodi
    - point_size: dimensione dei nodi
    - save_svg: salva anche come SVG (alta definizione)
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps["tab20"]
    colors = [cmap(i % 20) for i in range(20)]

    for idx, community in enumerate(communities_json["communities"]):
        member_ids = [
            int(member["name"].replace("node-", ""))
            for member in community["members"]
            if member["name"].startswith("node-") and int(member["name"].replace("node-", "")) in nodes
        ]

        if not member_ids:
            continue

        coords = [nodes[nid] for nid in member_ids]
        lats, lons = zip(*coords)
        ax.scatter(
            lons, lats,
            color=colors[idx % len(colors)],
            s=point_size,
            alpha=alpha,
            edgecolors='none',
            label=community["name"]
        )

    ax.set_title("Distribuzione delle comunità", fontsize=14)
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.set_aspect('equal', adjustable='box')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
    plt.tight_layout()
    if save_svg:
        plt.savefig("plot_communities.svg", format="svg")
        print("✅ Plot salvato come plot_communities.svg")
    plt.show()

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
            raw = json.load(f)
            nodes = {i: v for i, v in enumerate(raw.values())}
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

    plot_nodes(nodes)
    # Genera matrice delle distanze
    #route_matrix = generate_route_matrix_numpy(nodes)

    plot_clusters_overlap_2(nodes, json_overlap['centroids'], json_overlap['cluster_data'],)
    '''
    print("ho salvato overlap")
    plot_clusters_overlap_2(nodes, json_overlap['centroids'], json_overlap['cluster_data'],)
    print("ho plottato overlap")
    plot_clusters_html(nodes, json_overlap["cluster_data"], json_overlap["centroids"])
    print("ho plottato overlap")

    print("plotto slpa")
    plot_communities_2(nodes, json_original)

    plot_nodes_with_resource_size(
        nodes=nodes,
        scores=resource_exp_list,
        title="nodes_exp_plot"
    )

    plot_nodes_with_resource_size(
        nodes=nodes,
        scores=resource_log_list,
        title="nodes_log_plot"
    )

    plot_clusters_with_size_and_centroids(
        nodes=nodes,
        scores=resource_exp_list,
        cluster_json=json_resource_exp,
        title="resource_exp_plot"
    )

    plot_clusters_with_size_and_centroids(
        nodes=nodes,
        scores=resource_log_list,
        cluster_json=json_resource_log,
        title="resource_log_plot"
    )
    '''

if __name__ == "__main__":
    main()
