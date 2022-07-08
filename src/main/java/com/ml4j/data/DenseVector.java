package com.ml4j.data;

import com.ml4j.initializer.VectorUtils;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;

/**
 * @author: kexin
 * @date: 2022/6/23 21:44
 **/
public class DenseVector extends Tensor<float[]> implements IVector {
    private float[] data;

    public DenseVector() {
    }

    public DenseVector(int size) {
        this.data = new float[size];
    }

    public DenseVector(float[] data) {
        this.data = data;
    }

    @Override
    public DenseVector concat(Tensor vec0) {
        assert vec0 instanceof DenseVector;
        DenseVector vec = (DenseVector)vec0;
        float[] newData = new float[data.length + vec.data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        System.arraycopy(vec.data, 0, newData, data.length, vec.data.length);
        return new DenseVector(newData);
    }

    public DenseVector copy() {
        float[] newData = new float[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        return new DenseVector(newData);
    }

    @Override
    public Tensor valuesLike(float x) {
        float[] newData = new float[data.length];
        Arrays.fill(newData, x);
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

    public float reduce(float initValue, final BinaryOperator<Float> function) {
        float res = initValue;
        for (float x : data) {
            res = function.apply(res, x);
        }
        return res;
    }

    @Override
    public float sum() {
        return reduce(0, (a, b) -> a + b);
    }


    @Override
    public DenseVector elementWise(Tensor a, BiFunction<Float, Float, Float> function, boolean inPlace) {
        assert a instanceof DenseVector;
        DenseVector b = (DenseVector) a;
        assert data.length == b.data.length;
        float[] c;
        if (inPlace) {
            c = this.data;
        } else {
            c = new float[b.data.length];
        }
        for (int i = 0; i < data.length; i++) {
            c[i] = function.apply(data[i], b.data[i]);
        }
        if (inPlace) {
            return this;
        } else {
            return new DenseVector(c);
        }
    }


    @Override
    public DenseVector elementWise(Function<Float, Float> function, boolean inPlace) {
        float[] c;
        if (inPlace) {
            c = this.data;
        } else {
            c = new float[data.length];
        }
        for (int i = 0; i < data.length; i++) {
            c[i] = function.apply(data[i]);
        }
        if (inPlace) {
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

    /*
    public DenseVector abs(boolean inPlace){
        return elementWise(Math::abs, inPlace);
    }
    */
    /*
    public DenseVector elementWise(Function<Float, Float> function, boolean inPlace) {
        float[] c;
        if (inPlace) {
            c = this.data;
        } else {
            c = new float[data.length];
        }
        for (int i = 0; i < data.length; i++) {
            c[i] = function.apply(data[i]);
        }
        if (inPlace) {
            return this;
        } else {
            return new DenseVector(c);
        }
    }
    */
    /*
    public DenseVector minus(DenseVector vec, boolean inPlace) {
        return elementWise(vec, (a, b) -> a - b, inPlace);
    }

    public DenseVector add(DenseVector vec, boolean inPlace) {
        return elementWise(vec, (a, b) -> a + b, inPlace);
    }
    */

    /*
    public DenseVector multiply(float x, boolean inPlace) {
        return elementWise(a -> a * x, inPlace);
    }
    */

    /*
    public DenseVector multiply(DenseVector vec, boolean inPlace) {
        return (DenseVector) elementWise(vec, (a, b) -> a * b, inPlace);
    }
    */
}
