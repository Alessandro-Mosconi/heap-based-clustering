import json
import os
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Wedge, Patch
from scipy.spatial.distance import cdist

out_dir_unclustered = "real_data_unclustered"
out_dir_original = "real_data_original"
out_dir_overlap = "real_data_overlapped"
out_dir_resource = "real_data_resource"

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
    plt.savefig(out_dir_unclustered + "/" + 'node_plot.svg', format="svg")
    plt.show()


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
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir_unclustered + "/" + f"{to_snake_case(title)}.svg", format="svg")
    print(f"✅ Plot salvato in {title}.svg")
    plt.show()

def to_snake_case(text):
    # Sostituisce spazi e trattini con underscore
    text = re.sub(r'[\s\-]+', '_', text)
    # Inserisce underscore prima delle maiuscole (eccetto inizio parola)
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text)
    return text.lower()


'''
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
'''

def plot_clusters_overlap(
        nodes,
        centroids=None,
        cluster_data=None,
        title="Cluster with overlapping boundaries",
        figsize=(12, 10),
        alpha=0.7,
        point_size=1.85,
        wedge_scale=0.0008
):
    """
    Plots GPS nodes colored by cluster.
    Shared nodes are displayed as bicolored dots (wedge).
    Centroids are indicated with a black star.

    Modified to stretch vertically and remove white space.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cmap = plt.colormaps["tab20"]
    colors = [cmap(i % 20) for i in range(20)]

    shared_nodes_set = set()
    if cluster_data:
        for data in cluster_data:
            shared_nodes_set.update(data.get("shared_nodes", []))

    clustered_coords = {}
    plotted_nodes = set()
    legend_handles = []

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
                edgecolors='none'
            )
            legend_handles.append(
                Patch(color=colors[cluster_idx % 20], label=f"Cluster {cluster_idx + 1}")
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
            edgecolors='none'
        )
        legend_handles.append(Patch(color="gray", label="Non assegnati"))

    if centroids:
        centroid_coords = [nodes[c] for c in centroids if c in nodes]
        lat_c, lon_c = zip(*centroid_coords)
        ax.scatter(lon_c, lat_c, c='black', marker='*', s=10)

    # Calcola limiti con padding minimo
    all_lons = [lon for (lat, lon) in nodes.values()]
    all_lats = [lat for (lat, lon) in nodes.values()]
    ax.set_xlim(min(all_lons), max(all_lons))
    ax.set_ylim(min(all_lats), max(all_lats))
    ax.set_aspect('auto')  # Stretch verticale per usare tutto lo spazio

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    # Legenda esterna a destra
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.0,
        frameon=False
    )

    plt.savefig(out_dir_overlap + "/" + f"{to_snake_case(title)}.svg", format="svg", bbox_inches='tight')
    print(f"✅ Plot salvato in {title}.svg")
    plt.show()


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

from matplotlib.patches import Patch

def plot_communities(
    nodes,
    communities_json,
    figsize=(12, 10),
    alpha=0.8,
    point_size=4,
    cmap_name="tab20c",
    edge_color="black",
    edge_width=0.00,
    save_svg=True,
    title="SLPA community division",
    pad_pct=0.02
):

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    cmap = plt.get_cmap(cmap_name)
    comms = communities_json.get("communities", [])
    colors = [cmap(i % cmap.N) for i in range(len(comms))]

    all_lats, all_lons = [], []
    legend_handles = []

    for idx, community in enumerate(comms):
        member_ids = []
        for m in community.get("members", []):
            if m["name"].startswith("node-"):
                try:
                    nid = int(m["name"].split("node-")[1])
                except ValueError:
                    continue
                if nid in nodes:
                    member_ids.append(nid)
        if not member_ids:
            continue

        coords = [nodes[nid] for nid in member_ids]
        lats, lons = zip(*coords)
        all_lats.extend(lats)
        all_lons.extend(lons)

        ax.scatter(
            lons, lats,
            s=point_size,
            color=colors[idx],
            alpha=alpha,
            edgecolors=edge_color,
            linewidths=edge_width
        )

        name = community.get("name", f"Community {idx + 1}")
        legend_handles.append(Patch(color=colors[idx], label=name))

    if all_lats and all_lons:
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_pad = (lon_max - lon_min) * pad_pct
        lat_pad = (lat_max - lat_min) * pad_pct
        ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
        ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    # Legenda esterna uniforme con gli altri plot
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.0,
        frameon=False
    )

    plt.tight_layout()

    if save_svg:
        filename = f"{title.replace(' ', '_')}.svg"
        plt.savefig(out_dir_original + "/" + filename, format="svg", bbox_inches='tight')
        print(f"✅ Plot salvato in {filename}")

    plt.show()


def plot_clusters_with_size_and_centroids(
    nodes,
    scores,
    cluster_json,
    title="resource_exp_plot",
    figsize=(15, 12),
    min_size=5.0,
    max_size=80.0,
    alpha=0.6,
    edge_color="black",
    edge_width=0.4,
    centroid_marker="*",
    centroid_size=200,
    centroid_color="black",
    cmap_name="tab20c",
    pad_pct=0.02,
    save_svg=True
):

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    node_ids = np.array(list(nodes.keys()))
    scores = np.array(scores)

    # Normalizzazione score → dimensioni
    vmin, vmax = np.percentile(scores, 1), np.percentile(scores, 99)
    norm = (scores - vmin) / (vmax - vmin)
    sizes_map = min_size + norm * (max_size - min_size)
    node_id_to_size = dict(zip(node_ids, sizes_map))

    coords = {nid: nodes[nid] for nid in node_ids}
    clusters = cluster_json["cluster_data"]
    centroids = set(cluster_json.get("centroids", []))

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(clusters))]
    legend_handles = []

    all_lats, all_lons = [], []

    for idx, cluster_entry in enumerate(clusters):
        cluster_nodes = [nid for nid in cluster_entry["cluster"] if nid in coords]
        if not cluster_nodes:
            continue

        cluster_coords = np.array([coords[nid] for nid in cluster_nodes])
        lats, lons = cluster_coords[:, 0], cluster_coords[:, 1]
        all_lats.extend(lats)
        all_lons.extend(lons)

        sizes = np.array([node_id_to_size[nid] for nid in cluster_nodes])
        color = colors[idx]

        ax.scatter(
            lons, lats,
            s=sizes,
            color=color,
            alpha=alpha,
            edgecolors=edge_color,
            linewidths=edge_width
        )

        # Aggiungi entry per legenda
        legend_handles.append(mpatches.Patch(color=color, label=f"Cluster {idx + 1}"))

    # Disegna i centroidi
    for cid in centroids:
        if cid in coords:
            lat, lon = coords[cid]
            ax.scatter(
                lon, lat,
                s=centroid_size,
                color=centroid_color,
                marker=centroid_marker,
                edgecolors="white",
                linewidths=1.0,
                zorder=5
            )

    # Padding automatico
    if all_lats and all_lons:
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_pad = (lon_max - lon_min) * pad_pct
        lat_pad = (lat_max - lat_min) * pad_pct
        ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
        ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    # Legenda esterna a destra
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.0,
        frameon=False
    )

    plt.tight_layout()

    if save_svg:
        filename = f"{to_snake_case(title)}.svg"
        plt.savefig(out_dir_resource + "/" + filename, format="svg", bbox_inches="tight")
        print(f"✅ Plot salvato in {filename}")

    plt.show()



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


    print("Plot nodes")
    #plot_nodes(nodes)

    print("Plot nodes Resource exp")
    '''plot_nodes_with_resource_size(
        nodes=nodes,
        scores=resource_exp_list,
        title="Node distribution with exponential resource distribution",
    )'''

    print("Plot nodes Resource log")
    '''plot_nodes_with_resource_size(
        nodes=nodes,
        scores=resource_log_list,
        title="Node distribution with log-norm resource distribution"
    )'''

    print("Plot SLPA")
    plot_communities(nodes, json_original)

    print("Plot Overlap")
    #plot_clusters_overlap(nodes, json_overlap['centroids'], json_overlap['cluster_data'], )

    #print("Plot Overlap html")
    #plot_clusters_html(nodes, json_overlap["cluster_data"], json_overlap["centroids"])


    print("Plot Resource exp")
    '''plot_clusters_with_size_and_centroids(
        nodes=nodes,
        scores=resource_exp_list,
        cluster_json=json_resource_exp,
        title="Resource aware with exponential resource distribution"
    )'''

    print("Plot Resource log")
    '''plot_clusters_with_size_and_centroids(
        nodes=nodes,
        scores=resource_log_list,
        cluster_json=json_resource_log,
        title="Resource aware with log-norm resource distribution"
    )'''


if __name__ == "__main__":
    main()
