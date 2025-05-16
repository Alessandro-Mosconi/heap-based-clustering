package restserver.overlap.utils;

import com.google.gson.Gson;
import restserver.overlap.OverlappingClustering;
import restserver.partition.SimpleClusterData;

import java.util.*;

public class ClusterJsonExporter {

    public static String generateClusterDataJson(
            List<OverlappingClustering.ClusteringResult> results,
            List<Integer> centroids,
            long computation_time,
            long data_initialization_time
    ) {
        List<SimpleClusterData> simplified = new ArrayList<>();

        for (OverlappingClustering.ClusteringResult result : results) {
            simplified.add(new SimpleClusterData(result.cluster, result.exclusiveNodes, result.sharedNodes));
        }

        PartitionResultOverlappingCluster output = new PartitionResultOverlappingCluster(simplified, centroids, computation_time, data_initialization_time);
        return new Gson().toJson(output);
    }
}
