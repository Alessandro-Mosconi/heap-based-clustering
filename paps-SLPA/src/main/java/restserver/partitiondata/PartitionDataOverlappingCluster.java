package restserver.partitiondata;


import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

import java.util.List;

public class PartitionDataOverlappingCluster {

    private final PartitionParametersOverlappingCluster parameters;
    private final List<Host> hosts;

    @SerializedName("delay-matrix")
    private final DelayMatrix matrix;

    /**
     * Constructor
     * @param parameters
     * @param hosts
     * @param matrix
     */
    public PartitionDataOverlappingCluster(PartitionParametersOverlappingCluster parameters, List<Host> hosts, DelayMatrix matrix) {
        this.parameters = parameters;
        this.hosts = hosts;
        this.matrix = matrix;
    }

    public PartitionParametersOverlappingCluster getParameters() {
        return parameters;
    }

    public List<Host> getHosts() {
        return hosts;
    }

    public DelayMatrix getMatrix() {
        return matrix;
    }

    @Override
    public String toString() {
        return new Gson().toJson(this, PartitionDataOverlappingCluster.class);
    }
}

