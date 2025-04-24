package restserver.partition;

import com.google.gson.Gson;
import restserver.partitiondata.PartitionResultOverlappingCluster;

import java.util.*;

public class ClusterJsonExporter {

    public static String generateClusterDataJson(
            List<HeapClustering.ClusteringResult> results,
            List<Integer> centroids,
            long computation_time,
            long data_initialization_time
    ) {
        List<SimpleClusterData> simplified = new ArrayList<>();

        for (HeapClustering.ClusteringResult result : results) {
            simplified.add(new SimpleClusterData(result.cluster, result.exclusiveNodes, result.sharedNodes));
        }

        PartitionResultOverlappingCluster output = new PartitionResultOverlappingCluster(simplified, centroids, computation_time, data_initialization_time);
        return new Gson().toJson(output);
    }
}
