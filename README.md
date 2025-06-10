# heap-based-clustering

Collection of clustering algorithms applied to graphs and networks. The repository provides two ways to run them:

- **Java REST server** (`paps-SLPA`) to run the algorithms via HTTP
- **Python scripts** (`python_clustering`) for quick tests and visualisations
- **Sample datasets** and validation scripts (`test`)

## Structure

| Directory | Contents |
|-----------|----------|
| `paps-SLPA/` | Java REST server based on Spark |
| `python_clustering/` | Python scripts to experiment with the algorithms |
| `test/` | Example datasets and validation scripts |

```
heap-based-clustering/
├── paps-SLPA/
├── python_clustering/
└── test/
```

### 1. Java module `paps-SLPA`
The project uses Maven (see `pom.xml`) and provides a small REST server built on Spark. The main endpoints are summarised below:

| Endpoint | Method | Description |
|----------|-------|-------------|
| `/communities` | `POST` | traditional SLPA clustering |
| `/communities/overlap` | `POST` | clustering with shared nodes |
| `/communities/resource` | `POST` | resource-aware balanced clustering |

To start the Java server:

```bash
cd paps-SLPA
mvn package
java -jar target/paps-SLPA-rest-jar-with-dependencies.jar
```

The service exposes the endpoints on the default port `4567` with the prefix `/api`. The endpoints `/communities/overlap` and `/communities/resource` are the most experimental:

- **/communities/overlap**: runs the `OverlappingClustering` algorithm and returns the clusters highlighting exclusive and shared nodes.
- **/communities/resource**: uses `ResourceAwareClustering` to balance the number of nodes and the resources assigned to each cluster.

Use the file `paps-SLPA/src/main/java/restserver/example.json` as a reference for the input format.

### 2. Python scripts
The `python_clustering` folder contains educational implementations of the same algorithms, useful for quick tests without starting the Java server.

| Folder | Description |
|--------|-------------|
| `overlapping_clustering/` | linear and random versions of clustering with shared nodes |
| `resource_clustering/` | algorithms focused on resource balancing |

The scripts save the results as JSON files and PNG images. See the individual files for parameters and execution instructions.

### 3. Test data
The `test` folder holds synthetic and real datasets together with validation scripts.

| Directory | Description |
|-----------|-------------|
| `25x25`, `40x40`, `50x50_random` | small synthetic examples |
| `TAXI_DATASET` and `real_data_*` | real data and analysis scripts |

## Getting started
1. Inspect `example.json` to understand the basic request format.
2. Try the notebooks or Python scripts to familiarise yourself with the algorithm parameters.
3. Build the Java server and send requests to `/communities/overlap` and `/communities/resource` to test the results on a larger scale.
