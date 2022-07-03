package com.ml4j.initializer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;

import java.util.Arrays;

/**
 * @author: kexin
 * @date: 2022/6/25 17:04
 **/
public class ZeroInitializer extends Initializer {

    @Override
    synchronized public void init(Tensor v) {
        if(v instanceof DenseVector){
            float[] a = ((DenseVector)v).data();
            Arrays.fill(a,0);
        } else if (v instanceof DenseMatrix) {
            float[][] a = ((DenseMatrix)v).data();
            for (int i = 0; i < a.length; i++) {
                Arrays.fill(a[i],0);
            }
        }
    }
}
