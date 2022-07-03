package com.ml4j.regularizer;

import com.ml4j.data.Tensor;

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

   public abstract float computeLoss(Tensor input);

   public abstract Tensor computeGrad(Tensor input);
}
