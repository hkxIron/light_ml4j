package com.ml4j.math;

import com.ml4j.data.DenseVector;

/**
 * @author: kexin
 * @date: 2022/6/25 13:43
 **/
public abstract class ActivateFunction {
    public abstract float[] activate(float[] x);

    public abstract float[] gradient(float[] x);

    public DenseVector activate(DenseVector x, boolean inPlace) {
        float[] c = activate(x.data());
        if (inPlace) {
            ///System.arraycopy(c, 0, x.data(), 0, x.data().length);
            x.data(c);
            return x;
        } else {
            return new DenseVector(c);
        }
    }

    public DenseVector gradient(DenseVector x, boolean inPlace) {
        float[] c = gradient(x.data());
        if (inPlace) {
            x.data(c);
            //System.arraycopy(c, 0, x.data(), 0, x.data().length);
            return x;
        } else {
            return new DenseVector(c);
        }
    }
}
