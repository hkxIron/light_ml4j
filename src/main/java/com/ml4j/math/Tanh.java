package com.ml4j.math;

import static com.ml4j.math.FunctionUtils.*;

/**
 * @author: kexin
 * @date: 2022/6/24 22:44
 **/
public class Tanh extends ActivateFunction {
    @Override
    public float[] activate(float[] x) {
        float[] arr = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            arr[i] = tanh(x[i]);
        }
        return arr;
    }

    @Override
    public float[] gradient(float[] x) {
        float[] arr = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            arr[i] = dTanh(x[i]);
        }
        return arr;
    }
}
