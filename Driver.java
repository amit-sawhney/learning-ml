public class Driver {
    public static void main(String[] args) {
        Matrix first = new Matrix(3, 3);
        Matrix second = new Matrix(4,3);

        first.add(second);
        System.out.println(first);
    }
}
