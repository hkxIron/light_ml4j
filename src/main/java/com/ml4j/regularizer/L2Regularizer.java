package com.ml4j.regularizer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;

import static com.ml4j.data.VectorUtils.innerProduct;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Email: hukexin@xiaomi.com
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
     * @param matrix
     * @return
     */
    @Override
    public float computeLoss(DenseMatrix matrix) {
        float loss = 0;
        if (this.alpha <= 0) {
            return loss;
        }
        int[] shapes = matrix.getShape();
        for (int r = 0; r < shapes[0]; r++) {
            loss += innerProduct(matrix.data()[r], matrix.data()[r]);
        }
        return loss * alpha;
    }

    @Override
    public float computeLoss(DenseVector vector) {
        float loss = 0;
        if (this.alpha <= 0) {
            return loss;
        }
        loss = innerProduct(vector.data(), vector.data());
        return loss * alpha;
    }

    @Override
    public DenseMatrix computeGrad(DenseMatrix matrix) {
        if (this.alpha <= 0) {
            int[] shapes = matrix.getShape();
            return new DenseMatrix(shapes[0], shapes[1]);
        }
        return matrix.copy().multiply(alpha, false);
    }

    @Override
    public DenseVector computeGrad(DenseVector vector) {
        if (this.alpha <= 0) {
            int[] shapes = vector.getShape();
            return new DenseVector(shapes[0]);
        }
        return vector.copy().multiply(alpha, false);
    }
}
