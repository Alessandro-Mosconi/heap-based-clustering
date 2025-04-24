package restserver.partitiondata;

import com.google.gson.annotations.SerializedName;

public class DelayMatrix {

    @SerializedName("routes")
    private final double[][] routes;

    public DelayMatrix(double[][] routes) {
        this.routes = routes;
    }

    public double[][] getRoutes() {
        return routes;
    }
}
