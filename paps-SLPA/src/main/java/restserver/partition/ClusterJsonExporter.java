package restserver.partition;

import com.google.gson.Gson;
import restserver.partitiondata.PartitionResultOverlappingCluster;

import java.util.*;

public class ClusterJsonExporter {

    public static String generateClusterDataJson(
            List<HeapClustering.ClusteringResult> results,
            List<Integer> centroids,
            long elapsedTime
    ) {
        List<SimpleClusterData> simplified = new ArrayList<>();

        for (HeapClustering.ClusteringResult result : results) {
            simplified.add(new SimpleClusterData(result.cluster, result.exclusiveNodes, result.sharedNodes));
        }

        PartitionResultOverlappingCluster output = new PartitionResultOverlappingCluster(simplified, centroids, elapsedTime);
        return new Gson().toJson(output);
    }
}
