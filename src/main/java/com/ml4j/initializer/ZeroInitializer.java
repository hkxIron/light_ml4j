package com.ml4j.initializer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Initializer;

import java.util.Arrays;

import static com.ml4j.math.FunctionUtils.clip;

/**
 * @author: kexin
 * @date: 2022/6/25 17:04
 **/
public class ZeroInitializer extends Initializer {
    public ZeroInitializer() { }

    @Override
    synchronized public void init(DenseVector v) {
        float[] a = v.data();
        Arrays.fill(a,0);
    }

    @Override
    synchronized public void init(DenseMatrix x) {
        float[][] a = x.data();
        for (int i = 0; i < a.length; i++) {
            Arrays.fill(a[i],0);
        }
    }
}
