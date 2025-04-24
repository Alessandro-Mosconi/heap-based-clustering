package restserver.partitiondata;

import java.util.List;

public class PartitionResultClusterOnly {
    public List<SimpleClusterOnly> cluster_data;
    public List<Integer> centroids;
    public long computation_time;
    public long data_initialization_time;

    public PartitionResultClusterOnly(List<SimpleClusterOnly> cluster_data, List<Integer> centroids, long computation_time, long data_initialization_time) {
        this.cluster_data = cluster_data;
        this.centroids = centroids;
        this.computation_time = computation_time;
        this.data_initialization_time = data_initialization_time;
    }
}
