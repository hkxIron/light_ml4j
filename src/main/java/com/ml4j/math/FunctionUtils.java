package com.ml4j.math;

import static com.ml4j.initializer.VectorUtils.argMax;

/**
 * @author: kexin
 * @date: 2022/6/23 23:04
 **/
public class FunctionUtils {

    public static float clip(float x, float lower, float upper) {
        assert upper >= lower;
        if (x > upper) {
            x = upper;
        }
        if (x < lower) {
            x = lower;
        }
        return x;
    }

    public static float sigmoid(float x) {
        if (x >= 10) {
            return 1f;
        } else if (x <= -10) {
            return 0f;
        }
        double result = 1 / (1 + Math.pow(Math.E, -x));
        return (float) result;
    }

    /**
     * 公式： dsigmoid(input)= sigmoid(1-sigmoid)
     *
     * @param x
     * @return
     */
    public static float dSigmoid(float x) {
        float res = sigmoid(x);
        return res * (1 - res);
    }

    public static float relu(float x) {
        return x <= 0 ? 0 : x;
    }

    public static float dRelu(float x) {
        return x <= 0 ? 0 : 1;
    }

    /**
     * yi = exp(x_i)/sum(exp(x_k)) = exp(x_i-x0)/sum(exp(x_k-x0))
     * <p>
     * <p>
     * yi =exp(x_i)/sum(exp(x_k))
     * <p>
     * dy_i/dx_j
     * <p>
     * if(i==j){
     * yi(1-yi)
     * } else{
     * -yi*yi
     * }
     * <p>
     * 因此它的梯度一般有n^2项，所以一般并不单独求softmax的梯度，而是与CrossEntryLoss一起使用
     */
    public static float[] softmax(float[] x) {
        float[] arr = new float[x.length];
        float max = x[argMax(x)];
        // 为了数值稳定性，每个值减去最小值，并不影响最终结果
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            arr[i] = (float) Math.exp(x[i] - max);
            sum += arr[i];
        }

        for (int i = 0; i < x.length; i++) {
            arr[i] /= sum;
        }
        return arr;
    }

    /***
     * tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))
     *  = 2*sigmoid(2*x)-1
     *
     * tanh'(x) = 1 - (tanh(x))^2
     *
     */
    public static float tanh(float x) {
        return 2 * sigmoid(2 * x) - 1;
    }

    public static float dTanh(float x) {
        float t = tanh(x);
        return 1 - t * t;
    }
}
