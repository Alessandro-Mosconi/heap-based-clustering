package restserver.overlap.utils;

import com.google.gson.annotations.SerializedName;

public class PartitionParametersOverlappingCluster {
    @SerializedName("min_nodes_per_cluster")
    private final int minNodesPerCluster;
    @SerializedName("min_shared_nodes")
    private final int minSharedNodes;
    @SerializedName("min_exclusive_nodes")
    private final int minExclusiveNodes;

    public PartitionParametersOverlappingCluster(int minNodesPerCluster, int minSharedNodes, int minExclusiveNodes) {
        this.minNodesPerCluster = minNodesPerCluster;
        this.minSharedNodes = minSharedNodes;
        this.minExclusiveNodes = minExclusiveNodes;
    }

    public int getMinNodesPerCluster() {
        return minNodesPerCluster;
    }

    public int getMinSharedNodes() {
        return minSharedNodes;
    }

    public int getMinExclusiveNodes() {
        return minExclusiveNodes;
    }
}
