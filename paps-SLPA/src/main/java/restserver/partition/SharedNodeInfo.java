package restserver.partition;

import java.util.List;

public class SharedNodeInfo extends NodeInfo {
    public List<Integer> clusters;

    public SharedNodeInfo(int nodeId, double x, double y, List<Integer> clusters) {
        super(nodeId, x, y);
        this.clusters = clusters;
    }
}
