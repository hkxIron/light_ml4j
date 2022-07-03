package com.ml4j.network;

import com.ml4j.data.DenseVector;

import static com.ml4j.math.FunctionUtils.softmax;

/**
 * loss层的特点是没有参数
 *
 * @author: kexin
 * @date: 2022/6/25 16:04
 **/
public class SoftmaxWithCrossEntropyLoss extends Loss {

    /**
     * yi =softmax(xi)
     * <p>
     * CrossEntropyLoss = - sum_i(label_i*log(yi_i))
     *
     * @return
     */
    @Override
    public float computeLoss() {
        DenseVector x = this.getInput();
        DenseVector label = this.getLabel();
        assert x.data().length == label.data().length;
        int classNum = label.data().length;
        pred = predict();
        float loss = 0;
        for (int i = 0; i < classNum; i++) {
            if ((int) label.data()[i] == 1) {
                loss += -Math.log(pred.data()[i]);
            }
        }
        return loss;
    }

    @Override
    public DenseVector predict() {
        float[] pred = softmax(this.getInput().data());
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
        return (DenseVector) this.pred.minus(this.getLabel(), false);
    }
}
