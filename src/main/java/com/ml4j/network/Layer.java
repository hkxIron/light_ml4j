package com.ml4j.network;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Initializer;
import com.ml4j.optimizer.Optimizer;
import lombok.Getter;

/**
 * @author: kexin
 * @date: 2022/6/23 22:57
 **/
public interface Layer {
    DenseVector forward();
    DenseVector backward(DenseVector delta);
    void update(Optimizer optimizer); // 更新权重
    void initWeights(int inSize, Initializer initializer);
    void setInput(DenseVector x);
    int getOutSize();
    int getInSize();
    float getRegularizationLoss(); // 可选, 计算正则化loss
}
