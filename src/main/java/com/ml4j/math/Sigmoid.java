package com.ml4j.math;

import static com.ml4j.math.FunctionUtils.dSigmoid;
import static com.ml4j.math.FunctionUtils.sigmoid;

/**
 * @author: kexin
 * @date: 2022/6/25 13:44
 **/
public class Sigmoid extends ActivateFunction {
    @Override
    public float[] activate(float[] x) {
        float[] arr = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            arr[i] = sigmoid(x[i]);
        }
        return arr;
    }

    @Override
    public float[] gradient(float[] x) {
        float[] arr = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            arr[i] = dSigmoid(x[i]);
        }
        return arr;
    }
}
