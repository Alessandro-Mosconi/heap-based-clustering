package restserver;

import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import restserver.ordering.Utils;
import restserver.partition.ClusterJsonExporter;
import restserver.partition.Community;
import restserver.partition.HeapClustering;
import restserver.partition.SLPA;
import restserver.partitiondata.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static spark.Spark.post;

public class PartitionRoute {

    private static final Logger logger = LoggerFactory.getLogger(PartitionRoute.class);

    static void configureRoutes() {
        post("/communities", (request, response) -> {

            logger.info("new POST request to partitions");

            response.type("application/json");

            PartitionData data = new Gson().fromJson(request.body(), PartitionData.class);

            logger.info("Partition data:");
            logger.info(data.toString());
            SLPA slpa = new SLPA(data);

            PartitionParameters parameters = data.getParameters();

            long time = System.currentTimeMillis();
            List<Community> communities = slpa.computeCommunities(parameters.getIterations(), parameters.getProbabilityThreshold());

            if (communities.size() > 1) communities = Utils.orderCommunities(communities);

            Long elapsedTime = System.currentTimeMillis() - time;
            return new Gson().toJson(new PartitionResult(communities, elapsedTime), PartitionResult.class);
        });


        post("/communities/2", (request, response) -> {

            logger.info("new POST request to partitions");

            response.type("application/json");

            PartitionDataOverlappingCluster data = new Gson().fromJson(request.body(), PartitionDataOverlappingCluster.class);

            logger.info("Partition data:");
            logger.info(data.toString());

            PartitionParametersOverlappingCluster parameters = data.getParameters();

            long start = System.currentTimeMillis();

            HeapClustering.BalancedClusteringResult result = HeapClustering.balancedClustering(
                    convertFloatToDouble(data.getMatrix().getRoutes()),
                    parameters.getMinNodesPerCluster(),
                    parameters.getMinSharedNodes(),
                    parameters.getMinExclusiveNodes()
            );

            long elapsed = System.currentTimeMillis() - start;

            String json = ClusterJsonExporter.generateClusterDataJson(
                    result.clusters,
                    result.centroids,
                    elapsed
            );


            return json;
        });
    }
    public static double[][] convertFloatToDouble(float[][] input) {
        double[][] result = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            result[i] = new double[input[i].length];
            for (int j = 0; j < input[i].length; j++) {
                result[i][j] = input[i][j];
            }
        }
        return result;
    }

}
