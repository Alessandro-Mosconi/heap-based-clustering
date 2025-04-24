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

            long startTimeInitializationData = System.currentTimeMillis();
            PartitionData data = new Gson().fromJson(request.body(), PartitionData.class);

            logger.info("PartitionData initializated after {} ms", System.currentTimeMillis() - startTimeInitializationData);

            SLPA slpa = new SLPA(data);

            logger.info("SLPA initializated after {} ms", System.currentTimeMillis() - startTimeInitializationData);

            PartitionParameters parameters = data.getParameters();

            long elapsedTimeDataInitialization = System.currentTimeMillis() - startTimeInitializationData;

            long startTimeAlgorithm = System.currentTimeMillis();

            logger.info("Starting SLPA");
            List<Community> communities = slpa.computeCommunities(parameters.getIterations(), parameters.getProbabilityThreshold());

            if (communities.size() > 1) communities = Utils.orderCommunities(communities);

            long elapsedTime = System.currentTimeMillis() - startTimeAlgorithm;
            logger.info("Finished SLPA");
            return new Gson().toJson(new PartitionResult(communities, elapsedTime, elapsedTimeDataInitialization), PartitionResult.class);
        });


        post("/communities/2", (request, response) -> {

            logger.info("new POST request to partitions");

            response.type("application/json");

            long startTimeInitializationData = System.currentTimeMillis();
            PartitionDataOverlappingCluster data = new Gson().fromJson(request.body(), PartitionDataOverlappingCluster.class);

            logger.info("PartitionData initializated after {} ms", System.currentTimeMillis() - startTimeInitializationData);

            PartitionParametersOverlappingCluster parameters = data.getParameters();

            long elapsedTimeDataInitialization = System.currentTimeMillis() - startTimeInitializationData;

            long startTimeAlgorithm = System.currentTimeMillis();

            logger.info("Starting overlapping clustering");
            HeapClustering.BalancedClusteringResult result = HeapClustering.balancedClustering(
                    data.getMatrix().getRoutes(),
                    parameters.getMinNodesPerCluster(),
                    parameters.getMinSharedNodes(),
                    parameters.getMinExclusiveNodes()
            );

            logger.info("Starting overlapping clustering");
            long elapsed = System.currentTimeMillis() - startTimeAlgorithm;

            String json = ClusterJsonExporter.generateClusterDataJson(
                    result.clusters,
                    result.centroids,
                    elapsed,
                    elapsedTimeDataInitialization
            );


            return json;
        });


        post("/communities/3", (request, response) -> {

            logger.info("new POST request to /communities/3");

            response.type("application/json");
            long startTimeInitializationData = System.currentTimeMillis();

            PartitionDataResourceAware data = new Gson().fromJson(request.body(), PartitionDataResourceAware.class);
            double[][] routeMatrix = data.getMatrix().getRoutes();
            double[] resourceList = data.getResources().stream().mapToDouble(Double::doubleValue).toArray();
            int maxNodesPerCluster = data.getMaxNodesPerCluster();

            long elapsedTimeDataInitialization = System.currentTimeMillis() - startTimeInitializationData;

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
                    elapsed,
                    elapsedTimeDataInitialization
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
