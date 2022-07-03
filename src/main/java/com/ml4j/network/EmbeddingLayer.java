package com.ml4j.network;

import com.ml4j.data.DenseVector;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;

/**
 * @author: kexin
 * @date: 2022/7/3 17:44
 **/
public class EmbeddingLayer implements Layer {
    @Override
    public DenseVector forward() {
        return null;
    }

    @Override
    public DenseVector backward(DenseVector delta) {
        return null;
    }

    @Override
    public void update(Optimizer optimizer) {

    }

    @Override
    public void initWeights(int inSize, Initializer initializer) {

    }

    @Override
    public void setInput(DenseVector x) {

    }

    @Override
    public int getOutSize() {
        return 0;
    }

    @Override
    public int getInSize() {
        return 0;
    }

    @Override
    public float getRegularizationLoss() {
        return 0;
    }
}
