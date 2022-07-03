package com.ml4j.regularizer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;

import static com.ml4j.initializer.VectorUtils.abs;
import static com.ml4j.initializer.VectorUtils.sum;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Date: 22-6-28
 * Time: 上午10:22
 */
public class L1Regularizer extends Regularizer {
    public L1Regularizer(float alpha) {
        super(alpha);
    }

    /**
     * loss = |w|
     * dLoss/dw =
     * 1 if w>0
     * 0 if w=0
     * else -1;
     *
     * @param input
     * @return
     */
    @Override
    public float computeLoss(Tensor input) {
        float loss = 0;
        if (this.alpha <= 0) {
            return loss;
        }
        loss = input.abs(false).sum();
        return loss * alpha;
    }

    @Override
    public Tensor computeGrad(Tensor input) {
        if (this.alpha <= 0) {
            return input.valuesLike(0);
        }
        Tensor grad = input.sign(false)
                .multiply(alpha,true);
        return grad;
    }

    /*
    @Override
    public DenseMatrix computeGrad(DenseMatrix matrix) {
        if (this.alpha <= 0) {
            int[] shapes = matrix.getShape();
            return new DenseMatrix(shapes[0], shapes[1]);
        }
        return matrix.abs(false);
    }

    @Override
    public DenseVector computeGrad(DenseVector vector) {
        if (this.alpha <= 0) {
            int[] shapes = vector.getShape();
            return new DenseVector(shapes[0]);
        }
        return new DenseVector(abs(vector.data(), false))
                .multiply(alpha, true);
    }
    */
}
