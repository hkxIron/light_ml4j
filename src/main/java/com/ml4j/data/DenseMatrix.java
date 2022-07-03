package com.ml4j.data;

import lombok.extern.slf4j.Slf4j;

import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;

import static com.ml4j.initializer.VectorUtils.allEquals;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Date: 22-6-17
 * Time: 下午5:42
 */
@Slf4j
public class DenseMatrix implements Tensor<float[][]> {
    // int[][] arr = {{1,2,3},{3,4,2}};
    private float[][] data;

    public DenseMatrix() {
    }

    public DenseMatrix(int row, int col) {
        this.data = new float[row][col];
    }

    public DenseMatrix(float[][] data) {
        this.data = data;
    }

    public int[] getShape() {
        assert data != null && data[0] != null;
        return new int[]{data.length, data[0].length};
    }

    @Override
    public float[][] data() {
        return this.data;
    }

    public DenseMatrix copy() {
        int[] shape = getShape();
        int M = shape[0];
        int N = shape[1];
        float[][] newData = new float[M][N];

        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                newData[m][n] = data[m][n];
            }
        }
        return new DenseMatrix(newData);
    }

    @Override
    public boolean equalsInTolerance(Tensor vec, float eps) {
        boolean isMatrix = vec instanceof DenseMatrix;
        if (!isMatrix) {
            return false;
        }
        DenseMatrix mat = (DenseMatrix) vec;
        return this.elementWise(mat, (a, b) -> Math.abs(a - b), false).sum() < eps;
    }

    @Override
    public void data(float[][] floats) {
        this.data = floats;
    }

    public DenseMatrix multiply(DenseMatrix mat) {
        int[] shape = getShape();
        int[] matShape = mat.getShape();
        assert shape.length == 2;
        assert shape.length == matShape.length;
        assert shape[1] == matShape[0];

        int M = shape[0];
        int K = shape[1];
        int N = matShape[1];
        float[][] newData = new float[M][N];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += data[m][k] * mat.data[k][n];
                }
                newData[m][n] = sum;
            }
        }
        return new DenseMatrix(newData);
    }

    public DenseMatrix transpose(boolean inPlace) {
        int[] shape = getShape();
        int M = shape[0];
        int N = shape[1];

        float[][] newData = new float[N][M];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                newData[n][m] = data[m][n];
            }
        }
        if (inPlace) {
            data = newData;
            return this;
        } else {
            return new DenseMatrix(newData);
        }
    }

    public DenseVector multiply(DenseVector vec) {
        int[] shape = getShape();
        int[] vecShape = vec.getShape();
        assert shape[1] == vecShape[0];
        int M = shape[0];
        int N = shape[1];

        float[] dot = new float[M];
        for (int r = 0; r < M; r++) {
            dot[r] = new DenseVector(data[r]).innerProduct(vec);
        }
        return new DenseVector(dot);
    }

    public DenseMatrix add(DenseMatrix mat, boolean inPlace) {
        int[] shape = getShape();
        int[] matShape = mat.getShape();
        assert allEquals(shape, matShape);
        return this.elementWise(mat, (a, b) -> a + b, inPlace);
    }

    public DenseMatrix multiply(float x, boolean inPlace) {
        return elementWise((a)->a*x, inPlace);
    }

    public DenseMatrix abs(boolean inPlace) {
        return elementWise(Math::abs, inPlace);
    }

    public DenseMatrix elementWise(Function<Float, Float> func, boolean inPlace) {
        int[] shape = getShape();
        int M = shape[0];
        int N = shape[1];
        float[][] newData;
        if (inPlace) {
            newData = this.data;
        } else {
            newData = new float[M][N];
        }
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                newData[m][n] = func.apply(data[m][n]);
            }
        }
        if (inPlace) {
            return this;
        } else {
            return new DenseMatrix(newData);
        }
    }

    public float sum() {
        return reduceSum(0).sum();
    }

    public DenseVector reduceSum(int axis) {
        return reduce(axis, 0, (a, b) -> a + b);
    }

    /**
     * @param axis      which axis to reduce, 0 : reduce on row,  1: reduce on col
     * @param initValue
     * @param function
     * @return
     */
    public DenseVector reduce(int axis, float initValue, BinaryOperator<Float> function) {
        assert axis == 0 || axis == 1;
        int[] shape = getShape();
        int M = shape[0];
        int N = shape[1];
        float[] vals;
        if (axis == 0) {
            vals = new float[N];
            float res = initValue;
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    res = function.apply(res, data[m][n]);
                }
                vals[n] = res;
            }
        } else { // axis ==1
            vals = new float[M];
            float res = initValue;
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    res = function.apply(res, data[m][n]);
                }
                vals[m] = res;
            }
        }
        return new DenseVector(vals);
    }

    public DenseMatrix elementWise(DenseMatrix mat, BiFunction<Float, Float, Float> func, boolean inPlace) {
        int[] shape = getShape();
        int M = shape[0];
        int N = shape[1];
        assert allEquals(shape, mat.getShape());

        float[][] newData;
        if (inPlace) {
            newData = this.data;
        } else {
            newData = new float[M][N];
        }
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                newData[m][n] = func.apply(data[m][n], mat.data[m][n]);
            }
        }
        if (inPlace) {
            return this;
        } else {
            return new DenseMatrix(newData);
        }
    }

    /*
    public DenseVector copyVector(int row, int col) {
        int[] shape = getShape();
        assert shape.length == 2;
        int M = shape[0];
        int N = shape[1];
        if (row > -1) {
            // copy row
            float[] result = new float[N];
            for (int i = 0; i < N; i++) {
                result[i] = data[row][i];
            }
            return new DenseVector(result);
        } else if (col > -1) {
            // copy col
            float[] result = new float[M];
            for (int i = 0; i < M; i++) {
                result[i] = data[i][col];
            }
            return new DenseVector(result);
        } else {
            throw new RuntimeException("must specify at least one row or one column");
        }
    }
    */
}
