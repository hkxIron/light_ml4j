package com.ml4j.regularizer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 *
 * Date: 22-6-28
 * Time: 上午10:16
 */
public abstract class Regularizer {
   protected float alpha; // 权重系数

   public Regularizer(float alpha) {
      this.alpha = alpha;
   }

   public abstract float computeLoss(DenseMatrix matrix);
   public abstract float computeLoss(DenseVector vector);

   public abstract DenseMatrix computeGrad(DenseMatrix matrix);
   public abstract DenseVector computeGrad(DenseVector vector);
}
