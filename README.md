# heap-based-clustering

Repository sperimentale per algoritmi di clustering applicati a grafi e reti. Il progetto include:

* **Server REST Java** (cartella `paps-SLPA`) con endpoint per eseguire gli algoritmi via HTTP
* **Script Python** (cartella `python_clustering`) per studiare e visualizzare i diversi tipi di clustering
* Dati di esempio e test (cartella `test`)

## Struttura

```
heap-based-clustering/
├── paps-SLPA/               # implementazione server in Java
├── python_clustering/       # script ed esperimenti in Python
└── test/                    # dataset di prova e script di validazione
```

## 1. Modulo Java `paps-SLPA`

Il progetto è gestito con Maven (vedi `pom.xml`) e fornisce un piccolo server REST basato su Spark. Il server espone tre endpoint principali:

* `POST /communities` – clustering SLPA tradizionale
* `POST /communities/overlap` – clustering con nodi condivisi
* `POST /communities/resource` – clustering bilanciato in base alle risorse e al numero massimo di nodi per cluster

### Avvio del server

Per avviare il server:

```bash
cd paps-SLPA
mvn package
java -cp target/paps-SLPA-rest-jar-with-dependencies.jar restserver.PartitionRest
```

Il file `PartitionRest.java` rappresenta il punto di ingresso: avvia il server, ascolta le richieste in ingresso e restituisce i risultati dei clustering.

Una volta avviato, il server risponderà sotto `/api`. Gli endpoint `/communities/overlap` e `/communities/resource` sono quelli più sperimentali:

* **/communities/overlap**: applica l’algoritmo `OverlappingClustering` e restituisce i cluster con indicazione di nodi esclusivi e condivisi.
* **/communities/resource**: usa `ResourceAwareClustering` per bilanciare il numero di nodi e le risorse assegnate a ciascun cluster.

Il formato di input può essere preso come riferimento da `paps-SLPA/src/main/java/restserver/example.json`.

### Esempi di chiamate REST

#### SLPA tradizionale

```json
POST /api/communities
{
  "hosts": [
    { "name": "node-1", "labels": {} },
    { "name": "node-2", "labels": {} },
    { "name": "node-3", "labels": {} },
  ],
  "delay-matrix": {
    "routes": [
      [0.0, 10.0, 20.0],
      [10.0, 0.0, 15.0],
      [20.0, 15.0, 0.0]
    ]
  },
  "parameters": {
    "community-size": 2,
    "maximum-delay": 20,
    "iterations": 20,
    "probability-threshold": 30
  }
}
```

#### Clustering con nodi condivisi

```json
POST /api/communities/overlap
{
  "hosts": [
    { "name": "node-1", "labels": {} },
    { "name": "node-2", "labels": {} },
    { "name": "node-3", "labels": {} }
  ],
  "delay-matrix": {
    "routes": [
      [0.0, 10.0, 20.0],
      [10.0, 0.0, 15.0],
      [20.0, 15.0, 0.0]
    ]
  },
  "parameters": {
    "max_nodes_per_cluster": 2,
    "min_shared_nodes": 1,
    "min_exclusive_nodes": 1
  }
}
```

#### Clustering bilanciato per risorse

```json
POST /api/communities/resource
{
  "hosts": [
    { "name": "node-1", "labels": {} },
    { "name": "node-2", "labels": {} },
    { "name": "node-3", "labels": {} }
  ],
  "delay-matrix": {
    "routes": [
      [0.0, 10.0, 20.0],
      [10.0, 0.0, 15.0],
      [20.0, 15.0, 0.0]
    ]
  },
  "resources": [30, 50, 40],
  "max_number_nodes": 2
}
```

## 2. Script Python

La cartella `python_clustering` contiene implementazioni didattiche degli stessi algoritmi, utili per test veloci senza dover avviare il server Java.

* `overlapping_clustering/` – versioni lineari e random del clustering con nodi condivisi, con script per la generazione di grafici e resoconti JSON.
* `resource_clustering/` – algoritmi orientati al bilanciamento delle risorse, con output grafici e statistiche.

Gli script salvano i risultati sotto forma di file JSON e immagini PNG. Consultare i singoli file per i parametri e per le istruzioni di esecuzione.

## 3. Dati di test

La cartella `test` offre dataset sintetici e reali (ad esempio `TAXI_DATASET`) insieme a script di validazione e confronto delle metriche (`real_data_metrics.py`, `real_data_plot.py`).

## Suggerimenti per iniziare

1. Esaminare `example.json` per capire il formato base delle richieste.
2. Provare i notebook o gli script Python per familiarizzare con i parametri degli algoritmi.
3. Compilare il server Java ed effettuare richieste `POST` agli endpoint `/communities`, `/communities/overlap` e `/communities/resource`.
4. Analizzare la risposta JSON per ottenere la composizione dei cluster (nodi esclusivi, condivisi, utilizzo risorse).

Per dubbi o suggerimenti, apri una *issue* o una *pull request*.
