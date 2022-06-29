package com.ml4j.data;

import com.ml4j.initializer.VectorUtils;
import lombok.NoArgsConstructor;

/**
 * @author: kexin
 * @date: 2022/6/23 21:44
 **/
@NoArgsConstructor
public class DenseVector implements Tensor<float[]> {
    private float[] data;

    public DenseVector(int size) {
        this.data = new float[size];
    }

    public DenseVector(float[] data) {
        this.data = data;
    }

    public DenseVector copy() {
        float[] newData = new float[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        return new DenseVector(newData);
    }

    @Override
    public int[] getShape() {
        return new int[]{data.length};
    }

    @Override
    public float[] data() {
        return this.data;
    }

    @Override
    public void data(float[] floats) {
        this.data = floats;
    }

    public DenseVector minus(DenseVector b, boolean inPlace) {
        assert data.length == b.data.length;
        float[] c;
        if (inPlace) {
            c = this.data;
        } else {
            c = new float[b.data.length];
        }
        for (int i = 0; i < data.length; i++) {
            c[i] = data[i] - b.data[i];
        }
        if (inPlace) {
            //System.arraycopy(c, 0, data, 0, data.length);
            return this;
        } else {
            return new DenseVector(c);
        }
    }

    public DenseVector add(DenseVector b, boolean inPlace) {
        assert data.length == b.data.length;
        //float[] c = new float[b.data.length];
        float[] c;
        if (inPlace) {
            c = this.data;
        } else {
            c = new float[b.data.length];
        }
        for (int i = 0; i < data.length; i++) {
            c[i] = data[i] + b.data[i];
        }
        if (inPlace) {
            //System.arraycopy(c, 0, data, 0, data.length);
            return this;
        } else {
            return new DenseVector(c);
        }
    }

    public DenseVector multiply(float x, boolean inPlace) {
        float[] c;
        if (inPlace) {
            c = this.data;
        } else {
            c = new float[data.length];
        }
        for (int i = 0; i < data.length; i++) {
            c[i] = data[i] * x;
        }
        if (inPlace) {
            return this;
        } else {
            return new DenseVector(c);
        }
    }

    public DenseVector elementWiseMultiply(DenseVector b, boolean inPlace) {
        assert data.length == b.data.length;
        float[] c;
        if (inPlace) {
            c = this.data;
        } else {
            c = new float[b.data.length];
        }
        for (int i = 0; i < data.length; i++) {
            c[i] = data[i] * b.data[i];
        }
        if (inPlace) {
            //System.arraycopy(c, 0, data, 0, data.length);
            return this;
        } else {
            return new DenseVector(c);
        }
    }

    public float innerProduct(DenseVector b) {
        assert data.length == b.data.length;
        return VectorUtils.innerProduct(this.data, b.data);
    }

    /**
     * 向量外积
     * a:[M,1]
     * b:[1, N]
     * <p>
     * c= a * b
     * ->
     * c:[M, N]
     *
     * @param b
     * @return
     */
    public DenseMatrix outerProduct(DenseVector b) {
        int M = data.length;
        int N = b.data.length;
        float[][] c = new float[M][N];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                c[m][n] = data[m] * b.data[n];
            }
        }
        return new DenseMatrix(c);
    }
}
