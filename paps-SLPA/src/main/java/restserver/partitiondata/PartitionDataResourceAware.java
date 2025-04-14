package restserver.partitiondata;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class PartitionDataResourceAware {

    @SerializedName("delay-matrix")
    private final DelayMatrix matrix;

    private final List<Double> resources;

    @SerializedName("max_number_nodes") // 👈 usa questo invece di maxNodesPerCluster
    private final int maxNodesPerCluster;

    public PartitionDataResourceAware(DelayMatrix matrix, List<Double> resources, int maxNodesPerCluster) {
        this.matrix = matrix;
        this.resources = resources;
        this.maxNodesPerCluster = maxNodesPerCluster;
    }

    public DelayMatrix getMatrix() {
        return matrix;
    }

    public List<Double> getResources() {
        return resources;
    }

    public int getMaxNodesPerCluster() {
        return maxNodesPerCluster;
    }

    @Override
    public String toString() {
        return "PartitionDataResourceAware{" +
                "resources=" + resources +
                ", maxNodesPerCluster=" + maxNodesPerCluster +
                '}';
    }
}
