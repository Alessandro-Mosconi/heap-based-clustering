package restserver.partitiondata;

import restserver.partition.SimpleClusterData;

import java.util.List;

public class PartitionResultOverlappingCluster {
    public List<SimpleClusterData> cluster_data;
    public List<Integer> centroids;
    public long elapsedTime;

    public PartitionResultOverlappingCluster(
            List<SimpleClusterData> cluster_data,
            List<Integer> centroids,
            long elapsedTime
    ) {
        this.cluster_data = cluster_data;
        this.centroids = centroids;
        this.elapsedTime = elapsedTime;
    }
}
