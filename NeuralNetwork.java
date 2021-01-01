import java.util.List;
import java.util.ArrayList;

public class NeuralNetwork {
    private Matrix inputHiddenWeights, hiddenOutputWeights, hiddenBias, outputBias;
    private final double LEARNING_RATE = 0.01;

    /**
     * Constructor for Neural Network =
     * 
     * @param setInput  input to hidden weight matrix column size
     * @param setHidden input to hidden weight matrix row size and hidden to output
     *                  matrix column size
     * @param setOutput hidden to output matrix row size
     */
    public NeuralNetwork(int setInput, int setHidden, int setOutput) {
        inputHiddenWeights = new Matrix(setHidden, setInput);
        hiddenOutputWeights = new Matrix(setOutput, setHidden);

        hiddenBias = new Matrix(setHidden, 1);
        outputBias = new Matrix(setOutput, 1);
    }

    /**
     * Utilizes forward propogation
     * 
     * @param predictions
     * @return
     */
    public List<Double> predict(double[] predictions) {
        Matrix input = Matrix.fromArray(predictions);
        Matrix hidden = Matrix.multiply(inputHiddenWeights, input);
        hidden.add(hiddenBias);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(hiddenOutputWeights, hidden);
        output.add(outputBias);
        output.sigmoid();

        return output.toArrayList();
    }

    /**
     * Trains the neural network using backpropogation by calculating the errors in
     * prediction to adjust the matrices through gradients
     * 
     * @param trainPredictions
     * @param trainResults
     */
    public void train(double[] trainPredictions, double[] trainResults) {
        Matrix inputLayer = Matrix.fromArray(trainPredictions);
        Matrix hiddenLayer = Matrix.multiply(inputHiddenWeights, inputLayer);
        hiddenLayer.add(hiddenBias);
        hiddenLayer.sigmoid();

        Matrix output = Matrix.multiply(hiddenOutputWeights, hiddenLayer);
        output.add(outputBias);
        output.sigmoid();

        Matrix target = Matrix.fromArray(trainResults);

        Matrix error = Matrix.subtract(target, output);
        Matrix gradient = output.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(LEARNING_RATE);

        Matrix hiddenTranspose = Matrix.transpose(hiddenLayer);
        Matrix deltaHiddenOutput = Matrix.multiply(gradient, hiddenTranspose);

        hiddenOutputWeights.add(deltaHiddenOutput);
        outputBias.add(gradient);

        Matrix hiddenOutputWeightsTranspose = Matrix.transpose(hiddenOutputWeights);
        Matrix hiddenErrors = Matrix.multiply(hiddenOutputWeightsTranspose, error);

        Matrix hiddenGradient = hiddenLayer.dsigmoid();
        hiddenGradient.multiply(hiddenErrors);
        hiddenGradient.multiply(LEARNING_RATE);

        Matrix inputLayerTranspose = Matrix.transpose(inputLayer);
        Matrix deltaInputHiddenWeights = Matrix.multiply(hiddenGradient, inputLayerTranspose);

        inputHiddenWeights.add(deltaInputHiddenWeights);
        hiddenBias.add(hiddenGradient);
    }

    /**
     * Fits the neural network to a dataset
     * @param trainPredictionSamples the dependent variables in the training set
     * @param trainResultSamples the independent outputs in the training set
     * @param epochs the number of times for the model to see the data
     */
    public void fit(double[][] trainPredictionSamples, double[][] trainResultSamples, int epochs) {
        for (int i = 0; i < epochs; ++i) {
            int samplePrediction = (int) (Math.random() * trainPredictionSamples.length);
            this.train(trainPredictionSamples[samplePrediction], trainResultSamples[samplePrediction]);
        }
    }

}
