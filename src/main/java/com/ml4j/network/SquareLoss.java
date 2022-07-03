package com.ml4j.network;

import com.ml4j.data.DenseVector;

import static com.ml4j.math.FunctionUtils.sigmoid;

/**
 * loss层的特点是没有参数
 *
 * @author: kexin
 * @date: 2022/6/25 16:04
 **/
public class SquareLoss extends Loss {
    /**
     * yi =softmax(xi)
     * <p>
     * loss = (pi - yi)^2
     *
     * @return
     */
    @Override
    public float computeLoss() {
        DenseVector x = this.getInput();
        DenseVector label = this.getLabel();
        assert x.data().length == label.data().length;

        int size = x.data().length;
        pred = predict();
        float loss = 0;
        for (int i = 0; i < size; i++) {
            loss += Math.pow(label.data()[i] - x.data()[i], 2);
        }
        return loss;
    }

    @Override
    public DenseVector predict() {
        return this.getInput().copy();
    }

    /**
     * pi =softmax(xi)
     *
     * dLoss/dpi = 0.5*(pi- yi)
     *
     * @return
     */
    @Override
    public DenseVector computeGrad() {
        assert pred != null;
        return (DenseVector) this.pred.minus(this.getLabel(), false)
                .multiply(0.5f, true);
    }
}
