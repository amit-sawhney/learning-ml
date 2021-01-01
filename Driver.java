import java.util.ArrayList;
import java.util.List;

public class Driver {
    public static void main(String[] args) {
        double[][] X = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
        double[][] Y = { { 0 }, { 1 }, { 1 }, { 0 } };

        NeuralNetwork nn = new NeuralNetwork(2, 10, 1);
        nn.fit(X, Y, 50000);

        double[][] input = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
        List<Double> results = new ArrayList<Double>();
        for (double[] d : input) {
            List<Double> output = nn.predict(d);
            System.out.println(output.toString());
            double pred = output.get(0);
            if (pred > .5) {
                results.add(1.0);
                System.out.println("Expected Prediction: 1");
            } else {
                results.add(0.0);
                System.out.println("Expected Prediction: 0");
            }
        }

        int correct = 0;
        for (int i = 0; i < Y.length; i++) {
            if (results.get(i) == Y[i][0]) correct++;
        }

        System.out.println(correct / (1.0* results.size()) * 100 + "% accuracy");

    }
}
