import java.util.List;
import java.util.ArrayList;

/**
 * Java doesn't have a built in Matrix library as simple as numpy/pandas in
 * Python This class will need to be able to store the fundamental Data
 * Structure (DS) of Neural Networks (NN): Matrices FEATURES: - Addition -
 * Substraction - Transpose - Multiplication
 */
public class Matrix {
    private double[][] data;
    private int rows, cols;

    /**
     * Getter method for data.
     * 
     * @return data. The Matrix DS
     */
    public double[][] getData() {
        return data;
    }

    /**
     * Constructor that instantiates the 2D array to small random numbers. - This is
     * an expectation of the stochastic optimization algorithm which trains our
     * models (stochastic optimization) -
     * https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/#:~:text=The%20weights%20of%20artificial%20neural,model%2C%20called%20stochastic%20gradient%20descent.&text=About%20the%20need%20for%20nondeterministic%20and%20randomized%20algorithms%20for%20challenging%20problems.
     * 
     * @param rows the number of rows in our Matrix DS
     * @param cols the number of columns in our Matrix DS
     */
    public Matrix(int setRows, int setCols) {
        data = new double[setRows][setCols];
        this.rows = setRows;
        this.cols = setCols;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // random value between 1 and -1
                data[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    /**
     * Adds a scalar value to the entire Matrix
     * 
     * @param scalar value to add to each element of Matrix
     */
    public void add(double scalar) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] += scalar;
            }
        }
    }

    /**
     * Adds another matrix to our current Matrix DS
     * 
     * @param mat Matrix to add to current Matrix
     */
    public void add(Matrix mat) {
        if (rows != mat.rows || cols != mat.cols) {
            System.err.println("Invalid Matrix Shape (row, col). \n Expected" + rowColFormat(rows, cols) + "\n Received"
                    + rowColFormat(mat.rows, mat.cols));
            return;
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] += mat.data[i][j];
            }
        }
    }

    /**
     * Static Method that substracts two Matrices Returns null if inputs are invalid
     * 
     * @param first  the minuend matrix
     * @param second the subtrahend matrix
     * @return the difference Matrix between the first and second (first - second)
     */
    public static Matrix subtract(Matrix first, Matrix second) {
        if (first.rows != second.rows || first.cols != second.cols) {
            System.out.println("Dimensions" + rowColFormat(first.rows, first.cols) + "and "
                    + rowColFormat(second.rows, second.cols) + " are incompatible");
            return null;
        }
        Matrix difference = new Matrix(first.rows, first.cols);

        for (int i = 0; i < first.rows; ++i) {
            for (int j = 0; j < first.cols; ++j) {
                difference.data[i][j] = first.data[i][j] - second.data[i][j];
            }
        }

        return difference;
    }

    /**
     * Tranposes a Matrix
     * 
     * @param orig the original Matrix
     * @return the transpose of the original Matrix
     */
    public static Matrix transpose(Matrix orig) {
        Matrix transpose = new Matrix(orig.rows, orig.cols);

        for (int i = 0; i < orig.rows; ++i) {
            for (int j = 0; j < orig.cols; ++j) {
                transpose.data[i][j] = orig.data[j][i];
            }
        }

        return transpose;
    }

    /**
     * Static Method that multiplies a first and second method Returns null if input
     * is illegal
     * 
     * @param first  The first Matrix to multiply
     * @param second The second Matrix to multiply
     * @return The product of the two matrices
     */
    public static Matrix multiply(Matrix first, Matrix second) {
        if (first.cols != second.rows) {
            System.err.println("Inavlid Matrix Dimensions. Dimensions " + rowColFormat(first.rows, first.cols) + " and "
                    + rowColFormat(second.rows, second.cols) + " are not compatible");
            return null;
        }
        Matrix product = new Matrix(first.rows, second.cols);

        for (int i = 0; i < product.rows; ++i) {
            for (int j = 0; j < product.cols; ++j) {
                double sum = 0;
                for (int k = 0; k < first.cols; ++k) {
                    sum += second.data[k][j] * first.data[i][k];
                }
                product.data[i][j] = sum;
            }
        }
        return product;
    }

    /**
     * ELEMENT-WISE multiplication of a matrix
     * 
     * @param second the second matrix to multiply by
     */
    public void multiply(Matrix second) {
        if (rows != second.rows || cols != second.cols) {
            System.err.println(
                    "Invalid Dimensions. Dimensions should be equal for element-wise matrix multiplication. \n Expected: "
                            + rowColFormat(rows, cols) + " \n Received: " + rowColFormat(second.rows, second.cols));
            return;
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                this.data[i][j] *= second.data[i][j];
            }
        }
    }

    /**
     * Multiplies current Matrix by a scalar value
     * 
     * @param scalar the scalar to multiply by
     */
    public void multiply(double scalar) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                this.data[i][j] *= scalar;
            }
        }
    }

    /**
     * Acitvation function for Neural Network
     */
    public void sigmoid() {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                this.data[i][j] = sigmoidFunction(this.data[i][j]);
            }
        }
    }

    /**
     * Calculates the derivate sigmoid function on a Matrix. Used in the
     * Backpropogation calcuations. d/dx S(x) = S(x) * (1 - S(x))
     * 
     * @return the Matrix after inputed through the derivate of a sigmoid function
     * 
     */
    public Matrix dsigmoid() {
        Matrix transormedMatrix = new Matrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double sX = sigmoidFunction(this.data[i][j]);
                this.data[i][j] = sX * (1 - sX);
            }
        }
        return transormedMatrix;
    }

    /**
     * Converts array into a column vector
     * 
     * @param arr the array to convert
     * @return the column vector of the array argument
     */
    public static Matrix fromArray(double[] arr) {
        Matrix columnVector = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; ++i) {
            columnVector.data[i][0] = arr[i];
        }
        return columnVector;

    }

    /**
     * Converts a Matrix DS into a 1-dimensional ArrayList
     * 
     * @return the ArrayList version of the Matrix
     */
    public List<Double> toArrayList() {
        List<Double> transformDoubles = new ArrayList<Double>();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transformDoubles.add(data[i][j]);
            }
        }

        return transformDoubles;
    }

    /*
     * -----------------------------------------------------------------------
     * ----------------------------HELPER METHODS-----------------------------
     * -----------------------------------------------------------------------
     */

    /**
     * Returns sigmoid value of a number through a function S(x)
     * 
     * @param x the value to convert
     * @return the sigmoid value
     */
    private double sigmoidFunction(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /*
     * -----------------------------------------------------------------------
     * ------------------------FORMATTING METHODS-----------------------------
     * -----------------------------------------------------------------------
     */

    /**
     * Formats the Dimensions of a matrix into a string of the form (row, col)
     * 
     * @param row the number of rows of the matrix
     * @param col the number of columsn of the matrix
     * @return the string conversion into (row, col)
     */
    private static String rowColFormat(int row, int col) {
        return "(" + row + ", " + col + ")";
    }

    /**
     * Returns a printable string of a Matrix
     * 
     * @param mat the matrix to print
     * @return the String conversion of the matrix to print
     */
    public static String toString(Matrix mat) {
        String str = "----------------------- \n";

        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                str += mat.data[i][j] + " ";
            }
            str += "\n";
        }
        str += "----------------------- \n";
        str += "Dimensions" + rowColFormat(mat.rows, mat.cols) + "\n";
        str += "-----------------------";

        return str;
    }

    /**
     * Prints the current matrix object
     * 
     * @return the String conversion of the matrix to print
     */
    public String toString() {
        return toString(this);
    }
}
