package com.ml4j.initializer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;

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
    synchronized public void init(Tensor v) {
        if(v instanceof DenseVector){
            float[] a = ((DenseVector)v).data();
            for (int i = 0; i < a.length; i++) {
                a[i] = clip((float)rand.nextGaussian() * getStd() + getMean(), -getStd(), getStd());
            }
        } else if (v instanceof DenseMatrix) {
            float[][] a = ((DenseMatrix)v).data();
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    a[i][j] = clip((float)rand.nextGaussian() * getStd() + getMean(), -getStd(), getStd());
                }
            }
        }
    }
}
