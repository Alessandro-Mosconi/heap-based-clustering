package restserver.partition;

import java.util.List;

public class SimpleClusterData {
    public List<Integer> cluster;
    public List<Integer> exclusive_nodes;
    public List<Integer> shared_nodes;

    public SimpleClusterData(List<Integer> cluster, List<Integer> exclusive_nodes, List<Integer> shared_nodes) {
        this.cluster = cluster;
        this.exclusive_nodes = exclusive_nodes;
        this.shared_nodes = shared_nodes;
    }
}
