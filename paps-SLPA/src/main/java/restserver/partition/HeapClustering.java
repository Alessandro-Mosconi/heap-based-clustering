package restserver.partition;

import restserver.partitiondata.PartitionResultOverlappingCluster;

import java.util.*;

public class HeapClustering {
    public static class BalancedClusteringResult {
        public List<ClusteringResult> clusters;
        public List<Integer> centroids;

        public BalancedClusteringResult(List<ClusteringResult> clusters, List<Integer> centroids) {
            this.clusters = clusters;
            this.centroids = centroids;
        }
    }

    public static class ClusteringResult {
        public List<Integer> cluster;
        public List<Integer> exclusiveNodes;
        public List<Integer> sharedNodes;

        public ClusteringResult(List<Integer> cluster, List<Integer> exclusiveNodes, List<Integer> sharedNodes) {
            this.cluster = cluster;
            this.exclusiveNodes = exclusiveNodes;
            this.sharedNodes = sharedNodes;
        }
    }

    public static BalancedClusteringResult balancedClustering(
            double[][] routeMatrix,
            int minNodesPerCluster,
            int minSharedNodes,
            int minExclusiveNodes
    ) {
        int n = routeMatrix.length;
        int numClusters = n / minNodesPerCluster;
        int clusterSize = n / numClusters;

        if (minSharedNodes + minExclusiveNodes >= clusterSize) {
            throw new IllegalArgumentException("minSharedNodes + minExclusiveNodes must be less than cluster size.");
        }

        List<List<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < numClusters; i++) {
            clusters.add(new ArrayList<>());
        }

        boolean[] assigned = new boolean[n];
        List<Integer> centroids = new ArrayList<>();
        centroids.add(0);
        double[] cachedDists = new double[n];
        Arrays.fill(cachedDists, Double.POSITIVE_INFINITY);

        while (centroids.size() < numClusters) {
            double[] newDists = routeMatrix[centroids.get(centroids.size() - 1)];
            for (int i = 0; i < n; i++) {
                cachedDists[i] = Math.min(cachedDists[i], newDists[i]);
            }
            int newCentroid = maxIndex(cachedDists);
            centroids.add(newCentroid);
        }

        for (int i = 0; i < centroids.size(); i++) {
            int centroid = centroids.get(i);
            clusters.get(i).add(centroid);
            assigned[centroid] = true;
        }

        populateClusters(routeMatrix, clusters, centroids, clusterSize, assigned);

        for (int node = 0; node < n; node++) {
            if (!assigned[node]) {
                double minDist = Double.POSITIVE_INFINITY;
                int bestCluster = -1;
                for (int i = 0; i < centroids.size(); i++) {
                    double d = routeMatrix[node][centroids.get(i)];
                    if (d < minDist) {
                        minDist = d;
                        bestCluster = i;
                    }
                }
                clusters.get(bestCluster).add(node);
            }
        }

        Map<Integer, List<Integer>> sharedNodesMap = addSharedNodes(routeMatrix, clusters, centroids, minSharedNodes, minExclusiveNodes);

        List<ClusteringResult> results = new ArrayList<>();
        for (int i = 0; i < numClusters; i++) {
            List<Integer> cluster = clusters.get(i);
            Set<Integer> allShared = new HashSet<>();
            for (List<Integer> shared : sharedNodesMap.values()) {
                allShared.addAll(shared);
            }

            List<Integer> exclusive = new ArrayList<>();
            for (int node : cluster) {
                if (!allShared.contains(node)) {
                    exclusive.add(node);
                }
            }

            List<Integer> shared = sharedNodesMap.getOrDefault(i, new ArrayList<>());
            results.add(new ClusteringResult(cluster, exclusive, shared));
        }

        return new BalancedClusteringResult(results, centroids);
    }

    private static void populateClusters(
            double[][] routeMatrix,
            List<List<Integer>> clusters,
            List<Integer> centroids,
            int clusterSize,
            boolean[] assigned
    ) {
        int numClusters = clusters.size();
        List<PriorityQueue<NodeDist>> heaps = new ArrayList<>();
        for (int i = 0; i < numClusters; i++) {
            PriorityQueue<NodeDist> heap = new PriorityQueue<>();
            for (int j = 0; j < routeMatrix.length; j++) {
                if (!assigned[j]) {
                    heap.add(new NodeDist(routeMatrix[centroids.get(i)][j], j));
                }
            }
            heaps.add(heap);
        }

        for (int iter = 0; iter < routeMatrix.length; iter++) {
            for (int i = 0; i < numClusters; i++) {
                if (clusters.get(i).size() < clusterSize) {
                    while (!heaps.get(i).isEmpty()) {
                        NodeDist nd = heaps.get(i).poll();
                        if (!assigned[nd.node]) {
                            clusters.get(i).add(nd.node);
                            assigned[nd.node] = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    private static Map<Integer, List<Integer>> addSharedNodes(
            double[][] routeMatrix,
            List<List<Integer>> clusters,
            List<Integer> centroids,
            int minSharedNodes,
            int minExclusiveNodes
    ) {
        int n = routeMatrix.length;
        int numClusters = clusters.size();
        Map<Integer, List<Integer>> sharedNodes = new HashMap<>();

        double[][] distancesToCentroids = new double[n][numClusters];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < centroids.size(); j++) {
                distancesToCentroids[i][j] = routeMatrix[i][centroids.get(j)];
            }
        }

        for (int i = 0; i < numClusters; i++) {
            while (sharedNodes.getOrDefault(i, new ArrayList<>()).size() < minSharedNodes) {
                PriorityQueue<NodeDist> heap = new PriorityQueue<>();

                for (int j = 0; j < numClusters; j++) {
                    if (j == i) continue;

                    List<Integer> cluster = clusters.get(j);
                    Set<Integer> allShared = new HashSet<>();
                    for (List<Integer> s : sharedNodes.values()) allShared.addAll(s);

                    List<Integer> exclusiveNodes = new ArrayList<>();
                    for (int node : cluster) {
                        if (!allShared.contains(node)) {
                            exclusiveNodes.add(node);
                        }
                    }

                    if (exclusiveNodes.size() <= minExclusiveNodes) continue;

                    for (int node : exclusiveNodes) {
                        heap.add(new NodeDist(distancesToCentroids[node][i], node));
                    }
                }

                if (heap.isEmpty()) {
                    throw new IllegalStateException("Non ci sono abbastanza nodi esclusivi per soddisfare i vincoli di condivisione.");
                }

                NodeDist best = heap.poll();
                int originalCluster = -1;
                for (int j = 0; j < numClusters; j++) {
                    if (clusters.get(j).contains(best.node)) {
                        originalCluster = j;
                        break;
                    }
                }

                sharedNodes.computeIfAbsent(i, k -> new ArrayList<>()).add(best.node);
                sharedNodes.computeIfAbsent(originalCluster, k -> new ArrayList<>()).add(best.node);

                if (!clusters.get(i).contains(best.node)) {
                    clusters.get(i).add(best.node);
                }
            }
        }

        return sharedNodes;
    }

    private static int maxIndex(double[] array) {
        int idx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[idx]) {
                idx = i;
            }
        }
        return idx;
    }

    private static class NodeDist implements Comparable<NodeDist> {
        double distance;
        int node;

        NodeDist(double distance, int node) {
            this.distance = distance;
            this.node = node;
        }

        @Override
        public int compareTo(NodeDist other) {
            return Double.compare(this.distance, other.distance);
        }
    }


}
