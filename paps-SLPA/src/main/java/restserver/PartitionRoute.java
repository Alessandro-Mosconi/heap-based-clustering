package restserver;

import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import restserver.ordering.Utils;
import restserver.partition.*;
import restserver.partitiondata.*;
import restserver.partitiondata.SimpleClusterOnly;

import java.util.*;

import static spark.Spark.post;

public class PartitionRoute {

    private static final Logger logger = LoggerFactory.getLogger(PartitionRoute.class);

    static void configureRoutes() {
        post("/communities", (request, response) -> {

            logger.info("new POST request to partitions");

            response.type("application/json");

            PartitionData data = new Gson().fromJson(request.body(), PartitionData.class);

            SLPA slpa = new SLPA(data);

            PartitionParameters parameters = data.getParameters();

            long time = System.currentTimeMillis();

            logger.info("Starting SLPA");
            List<Community> communities = slpa.computeCommunities(parameters.getIterations(), parameters.getProbabilityThreshold());

            if (communities.size() > 1) communities = Utils.orderCommunities(communities);

            Long elapsedTime = System.currentTimeMillis() - time;
            logger.info("Finished SLPA");
            return new Gson().toJson(new PartitionResult(communities, elapsedTime), PartitionResult.class);
        });


        post("/communities/2", (request, response) -> {

            logger.info("new POST request to partitions");

            response.type("application/json");

            PartitionDataOverlappingCluster data = new Gson().fromJson(request.body(), PartitionDataOverlappingCluster.class);

            logger.info("Data received");
            PartitionParametersOverlappingCluster parameters = data.getParameters();

            long start = System.currentTimeMillis();

            logger.info("Starting overlapping clustering");
            HeapClustering.BalancedClusteringResult result = HeapClustering.balancedClustering(
                    convertFloatToDouble(data.getMatrix().getRoutes()),
                    parameters.getMinNodesPerCluster(),
                    parameters.getMinSharedNodes(),
                    parameters.getMinExclusiveNodes()
            );

            logger.info("Starting overlapping clustering");
            long elapsed = System.currentTimeMillis() - start;

            String json = ClusterJsonExporter.generateClusterDataJson(
                    result.clusters,
                    result.centroids,
                    elapsed
            );


            return json;
        });
        post("/communities/3", (request, response) -> {

            logger.info("new POST request to /communities/3");

            response.type("application/json");

            PartitionDataResourceAware data = new Gson().fromJson(request.body(), PartitionDataResourceAware.class);
            double[][] routeMatrix = convertFloatToDouble(data.getMatrix().getRoutes());
            double[] resourceList = data.getResources().stream().mapToDouble(Double::doubleValue).toArray();
            int maxNodesPerCluster = data.getMaxNodesPerCluster();

            long start = System.currentTimeMillis();

            logger.info("Starting resource clustering");
            ResourceAwareClustering.ClusteringOutput result = ResourceAwareClustering.balancedClusteringByNodesAndResources(
                    routeMatrix,
                    resourceList,
                    maxNodesPerCluster,
                    50,       // numero di iterazioni T
                    1.1       // tolleranza
            );

            logger.info("Finished resource clustering");
            long elapsed = System.currentTimeMillis() - start;

            List<SimpleClusterOnly> clusterData = new ArrayList<>();
            for (ResourceAwareClustering.ClusterResult cluster : result.clusterData) {
                clusterData.add(new SimpleClusterOnly(cluster.cluster));
            }

            PartitionResultClusterOnly responseObj = new PartitionResultClusterOnly(
                    clusterData,
                    result.centroids,
                    elapsed
            );

            return new Gson().toJson(responseObj);
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
