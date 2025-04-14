package restserver.partition;

import java.util.*;

public class ResourceAwareClustering {

    public static class ClusterResult {
        public List<Integer> cluster;
        public double totalResources;
        public int numNodes;

        public ClusterResult(List<Integer> cluster, double totalResources, int numNodes) {
            this.cluster = cluster;
            this.totalResources = totalResources;
            this.numNodes = numNodes;
        }
    }

    public static class ClusteringOutput {
        public List<ClusterResult> clusterData;
        public List<Integer> centroids;

        public ClusteringOutput(List<ClusterResult> clusterData, List<Integer> centroids) {
            this.clusterData = clusterData;
            this.centroids = centroids;
        }
    }

    public static ClusteringOutput balancedClusteringByNodesAndResources(
            double[][] routeMatrix,
            double[] resourceList,
            int maxNodesPerCluster,
            int T,
            double tolerance
    ) {
        int n = routeMatrix.length;
        int numClusters = Math.max(2, (int) Math.ceil((double) n / maxNodesPerCluster));
        double totalResources = Arrays.stream(resourceList).sum();
        double avgResources = totalResources / numClusters;

        List<Integer> centroids = new ArrayList<>();
        centroids.add(new Random().nextInt(n));
        for (int i = 1; i < numClusters; i++) {
            double[] minDists = new double[n];
            Arrays.fill(minDists, Double.MAX_VALUE);
            for (int c : centroids) {
                for (int j = 0; j < n; j++) {
                    minDists[j] = Math.min(minDists[j], routeMatrix[c][j]);
                }
            }
            int nextCentroid = argMax(minDists);
            centroids.add(nextCentroid);
        }

        boolean[] assigned = new boolean[n];
        List<List<Integer>> clusters = new ArrayList<>();
        double[] clusterResources = new double[numClusters];
        int[] clusterSizes = new int[numClusters];

        List<PriorityQueue<NodeDist>> heaps = new ArrayList<>();

        for (int i = 0; i < numClusters; i++) {
            clusters.add(new ArrayList<>());
            PriorityQueue<NodeDist> heap = new PriorityQueue<>();
            int c = centroids.get(i);
            for (int j = 0; j < n; j++) {
                if (j != c) heap.add(new NodeDist(routeMatrix[c][j], j));
            }
            heaps.add(heap);
            clusters.get(i).add(c);
            assigned[c] = true;
            clusterResources[i] += resourceList[c];
            clusterSizes[i]++;
        }

        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < numClusters; i++) {
                while (!heaps.get(i).isEmpty()) {
                    int node = heaps.get(i).poll().node;
                    if (assigned[node]) continue;
                    double projectedRes = clusterResources[i] + resourceList[node];
                    int projectedSize = clusterSizes[i] + 1;
                    if (projectedSize <= maxNodesPerCluster && projectedRes <= avgResources * 1.1) {
                        clusters.get(i).add(node);
                        assigned[node] = true;
                        clusterResources[i] = projectedRes;
                        clusterSizes[i] = projectedSize;
                        changed = true;
                        break;
                    }
                }
            }
        }

        for (int node = 0; node < n; node++) {
            if (!assigned[node]) {
                List<NodeDist> distances = new ArrayList<>();
                for (int i = 0; i < numClusters; i++) {
                    distances.add(new NodeDist(routeMatrix[node][centroids.get(i)], i));
                }
                distances.sort(Comparator.naturalOrder());
                for (NodeDist d : distances) {
                    int i = d.node;
                    if (clusterSizes[i] < maxNodesPerCluster) {
                        clusters.get(i).add(node);
                        clusterResources[i] += resourceList[node];
                        clusterSizes[i]++;
                        assigned[node] = true;
                        break;
                    }
                }
            }
        }

        for (int t = 0; t < T; t++) {
            for (int i = 0; i < numClusters; i++) {
                if (clusters.get(i).size() <= 1) continue;
                int centroid = centroids.get(i);
                List<NodeDist> farthest = new ArrayList<>();
                for (int node : clusters.get(i)) {
                    if (node != centroid) {
                        farthest.add(new NodeDist(routeMatrix[centroid][node], node));
                    }
                }
                if (farthest.isEmpty()) continue;
                farthest.sort((a, b) -> Double.compare(b.distance, a.distance));
                int nodeToMove = farthest.get(0).node;

                List<NodeDist> candidates = new ArrayList<>();
                for (int j = 0; j < numClusters; j++) {
                    if (j == i) continue;
                    if (clusterSizes[j] >= maxNodesPerCluster * tolerance) continue;
                    if (clusterResources[j] >= clusterResources[i]) continue;
                    double projRes = clusterResources[j] + resourceList[nodeToMove];
                    if (projRes > avgResources * tolerance) continue;
                    double dist = routeMatrix[nodeToMove][centroids.get(j)];
                    candidates.add(new NodeDist(dist, j));
                }

                if (!candidates.isEmpty()) {
                    candidates.sort(Comparator.naturalOrder());
                    int bestJ = candidates.get(0).node;

                    clusters.get(i).remove((Integer) nodeToMove);
                    clusters.get(bestJ).add(nodeToMove);

                    clusterResources[i] -= resourceList[nodeToMove];
                    clusterSizes[i]--;

                    clusterResources[bestJ] += resourceList[nodeToMove];
                    clusterSizes[bestJ]++;
                }
            }
        }

        List<ClusterResult> result = new ArrayList<>();
        for (int i = 0; i < numClusters; i++) {
            result.add(new ClusterResult(clusters.get(i), clusterResources[i], clusterSizes[i]));
        }

        return new ClusteringOutput(result, centroids);
    }

    private static int argMax(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
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
