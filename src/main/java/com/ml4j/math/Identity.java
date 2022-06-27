package com.ml4j.math;

import java.util.Arrays;

import static com.ml4j.math.FunctionUtils.dRelu;
import static com.ml4j.math.FunctionUtils.relu;

/**
 * @author: kexin
 * @date: 2022/6/25 13:44
 **/
public class Identity extends ActivateFunction {
    @Override
    public float[] activate(float[] x) {
        float[] arr = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            arr[i] = x[i];
        }
        return arr;
    }

    @Override
    public float[] gradient(float[] x) {
        float[] arr = new float[x.length];
        Arrays.fill(arr, 1);
        return arr;
    }
}
