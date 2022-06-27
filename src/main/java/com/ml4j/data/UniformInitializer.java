package com.ml4j.data;

/**
 * @author: kexin
 * @date: 2022/6/25 17:04
 **/
public class UniformInitializer extends Initializer {
    public UniformInitializer(float mean, float std) {
        super(mean, std);
    }

    public UniformInitializer() {
        this(0, 1);
    }

    @Override
    public void init(DenseVector v) {
        float[] a = v.data();
        for (int i = 0; i < a.length; i++) {
            a[i] = rand.nextFloat() * getStd() + getMean();
        }
    }

    @Override
    public void init(DenseMatrix x) {
        float[][] a = x.data();
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] = rand.nextFloat() * getStd() + getMean();
            }
        }
    }
}
