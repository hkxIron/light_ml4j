package com.ml4j.network;

import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;

/**
 * @author: kexin
 * @date: 2022/6/23 22:57
 **/
public interface ILayer {
    DenseVector forward();
    DenseVector backward(DenseVector delta);
    void update(Optimizer optimizer); // 更新权重
    void initWeights(Initializer initializer);
    void setInput(Tensor x);
    int getOutSize();
    void setOutSize(int size);
    void setInSize(int size);
    int getInSize();
    float getRegularizationLoss(); // 可选, 计算正则化loss
}
