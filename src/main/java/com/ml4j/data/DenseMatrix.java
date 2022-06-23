package com.ml4j.data;

import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Date: 22-6-17
 * Time: 下午5:42
 */
@Slf4j
@NoArgsConstructor
public class DenseMatrix implements Tensor<float[][]> {
    // int[][] arr = {{1,2,3},{3,4,2}};
    private float[][] data;

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

    public DenseMatrix multiply(DenseVector vec) {
        int[] shape = getShape();
        int[] vecShape = vec.getShape();
        assert shape.length == 2;
        assert shape[1] == vecShape[0];
        int N = shape[1];

        float[][] temp = new float[N][1];
        for (int i = 0; i < N; i++) {
            temp[i][0] = vec.data()[i];
        }
        return this.multiply(new DenseMatrix(temp));
    }
    public DenseVector multiplyToVector(DenseVector vec) {
        return multiply(vec).copyVector(-1, 0);
    }

    /**
     * @param row
     * @param col
     * @return
     */
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
}
