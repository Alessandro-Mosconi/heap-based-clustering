package restserver.partitiondata;

import restserver.partition.Community;

import java.util.List;

public class PartitionResult {
    private final List<Community> communities;
    private final long computationTime;

    public PartitionResult(List<Community> communities, long computationTime) {
        this.communities = communities;
        this.computationTime = computationTime;
    }
}
