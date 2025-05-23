package restserver.partitiondata;

import restserver.partition.Community;

import java.util.List;

public class PartitionResult {
    private final List<Community> communities;
    private final long computation_time;
    private final long data_initialization_time;

    public PartitionResult(List<Community> communities, long computation_time, long data_initialization_time) {
        this.communities = communities;
        this.computation_time = computation_time;
        this.data_initialization_time = data_initialization_time;
    }
}
