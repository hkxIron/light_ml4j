package com.ml4j.data;

import static com.ml4j.math.FunctionUtils.clip;

/**
 * @author: kexin
 * @date: 2022/6/25 17:04
 **/
public class TruncatedNormalInitializer extends Initializer {
    public TruncatedNormalInitializer(float mean, float std) {
        super(mean, std);
    }

    public TruncatedNormalInitializer() {
        this(0, 1);
    }

    @Override
    synchronized public void init(DenseVector v) {
        float[] a = v.data();
        for (int i = 0; i < a.length; i++) {
            a[i] = clip((float)rand.nextGaussian() * getStd() + getMean(), -getStd(), getStd());
        }
    }

    @Override
    synchronized public void init(DenseMatrix x) {
        float[][] a = x.data();
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] = clip((float)rand.nextGaussian() * getStd() + getMean(), -getStd(), getStd());
            }
        }
    }
}
