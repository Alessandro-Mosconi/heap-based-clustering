package restserver.overlap.utils;

import restserver.partition.SimpleClusterData;

import java.util.List;

public class PartitionResultOverlappingCluster {
    public List<SimpleClusterData> cluster_data;
    public List<Integer> centroids;
    public long computation_time;
    public long data_initialization_time;

    public PartitionResultOverlappingCluster(
            List<SimpleClusterData> cluster_data,
            List<Integer> centroids,
            long computation_time,
            long data_initialization_time
    ) {
        this.cluster_data = cluster_data;
        this.centroids = centroids;
        this.computation_time = computation_time;
        this.data_initialization_time = data_initialization_time;
    }
}
