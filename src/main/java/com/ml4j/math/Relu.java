package com.ml4j.math;

import static com.ml4j.math.FunctionUtils.*;

/**
 * @author: kexin
 * @date: 2022/6/25 13:44
 **/
public class Relu extends ActivateFunction {
    @Override
    public float[] activate(float[] x) {
        float[] arr = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            arr[i] = relu(x[i]);
        }
        return arr;
    }

    @Override
    public float[] gradient(float[] x) {
        float[] arr = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            arr[i] = dRelu(x[i]);
        }
        return arr;
    }
}
