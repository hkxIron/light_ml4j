package com.ml4j.data;

import java.util.Random;

/**
 * @author: kexin
 * @date: 2022/6/25 17:04
 **/
public class NormalInitializer extends Initializer {
    public NormalInitializer(float mean, float std) {
        super(mean, std);
    }

    public NormalInitializer() {
        this(0, 1);
    }

    @Override
    synchronized public void init(DenseVector v) {
        float[] a = v.data();
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) rand.nextGaussian() * getStd() + getMean();
        }
    }

    @Override
    synchronized public void init(DenseMatrix x) {
        float[][] a = x.data();
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; i++) {
                a[i][j] = (float) rand.nextGaussian() * getStd() + getMean();
            }
        }
    }
}
