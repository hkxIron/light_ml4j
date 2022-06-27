package com.ml4j.network;

import com.ml4j.data.DenseVector;

import static com.ml4j.math.FunctionUtils.sigmoid;
import static com.ml4j.math.FunctionUtils.softmax;

/**
 * loss层的特点是没有参数
 *
 * @author: kexin
 * @date: 2022/6/25 16:04
 **/
public class BinaryLoigitWithCrossEntropyLoss extends Loss {

    /**
     * yi =softmax(xi)
     * <p>
     * CrossEntropyLoss = (-1) [yi log(pi) + (1-yi) log(1-pi)]
     *
     * @return
     */
    @Override
    public float computeLoss() {
        DenseVector x = this.getInput();
        DenseVector label = this.getLabel();
        assert x.data().length == label.data().length;

        pred = predict();
        float loss = 0;
        if ((int) label.data()[0] == 1) {
            loss = -(float) Math.log(pred.data()[0]);
        } else {
            loss = -(float) Math.log(1 - pred.data()[0]);
        }
        return loss;
    }

    @Override
    public DenseVector predict() {
        float[] pred = {sigmoid(this.getInput().data()[0])};
        return new DenseVector(pred);
    }

    /**
     * yi =softmax(xi)
     * <p>
     * i==j:
     * grad = yi - yi^yi
     * <p>
     * i!=j:
     * grad = -yi^yi
     * <p>
     * <p>
     * CrossEntropyLoss = - sum_i(label_i*log(yi_i))
     * <p>
     * dLoss/dyi = yi - label_i
     *
     * @return
     */
    @Override
    public DenseVector computeGrad() {
        assert pred != null;
        return this.pred.minus(this.getLabel(), false);
    }
}
