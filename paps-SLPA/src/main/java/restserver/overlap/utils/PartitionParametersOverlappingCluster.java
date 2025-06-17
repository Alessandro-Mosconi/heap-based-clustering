package restserver.overlap.utils;

import com.google.gson.annotations.SerializedName;

public class PartitionParametersOverlappingCluster {
    @SerializedName("max_nodes_per_cluster")
    private final int maxNodesPerCluster;
    @SerializedName("min_shared_nodes")
    private final int minSharedNodes;
    @SerializedName("min_exclusive_nodes")
    private final int minExclusiveNodes;

    public PartitionParametersOverlappingCluster(int minNodesPerCluster, int minSharedNodes, int minExclusiveNodes) {
        this.maxNodesPerCluster = minNodesPerCluster;
        this.minSharedNodes = minSharedNodes;
        this.minExclusiveNodes = minExclusiveNodes;
    }

    public int getMaxNodesPerCluster() {
        return maxNodesPerCluster;
    }

    public int getMinSharedNodes() {
        return minSharedNodes;
    }

    public int getMinExclusiveNodes() {
        return minExclusiveNodes;
    }
}
