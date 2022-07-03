package com.ml4j.regularizer;

import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 *
 * Date: 22-6-28
 * Time: 上午10:22
 */
public class L2Regularizer extends Regularizer {
    public L2Regularizer(float alpha) {
        super(alpha);
    }

    /**
     * loss = w^2
     * dLoss/dw = 2*w
     *
     * @param input
     * @return
     */
    @Override
    public float computeLoss(Tensor input) {
        if (this.alpha <= 0) {
            return 0;
        }
        return input.pow(2f, false).sum() * alpha;
    }

    @Override
    public Tensor computeGrad(Tensor input) {
        if (this.alpha <= 0) {
            return input.valuesLike(0);
        }
        return input.copy().multiply(2 * alpha, true);
    }
}
