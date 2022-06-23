package com.ml4j.math;

/**
 * @author: kexin
 * @date: 2022/6/23 23:04
 **/
public class Functions {
    public static float sigmoid(float x) {
        double result = 1 / (1 + Math.pow(Math.E, -x));
        return (float) result;
    }

    /**
     * 公式： dsigmoid(x)= sigmoid(1-sigmoid)
     *
     * @param x
     * @return
     */
    public static float dSigmoid(float x) {
        float res = sigmoid(x);
        return res * (1 - res);
    }

    public static float relu(float x) {
        return x <= 0 ? 0: x;
    }

    public static float dRelu(float x) {
        return x <= 0 ? 0: 1;
    }
}
