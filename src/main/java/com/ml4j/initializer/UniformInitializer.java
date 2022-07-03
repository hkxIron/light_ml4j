package com.ml4j.initializer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;

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
    synchronized public void init(Tensor v) {
        if(v instanceof DenseVector){
            float[] a = ((DenseVector)v).data();
            for (int i = 0; i < a.length; i++) {
                a[i] = rand.nextFloat() * getStd() + getMean();
            }
        } else if (v instanceof DenseMatrix) {
            float[][] a = ((DenseMatrix)v).data();
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    a[i][j] = rand.nextFloat() * getStd() + getMean();
                }
            }
        }
    }
}
