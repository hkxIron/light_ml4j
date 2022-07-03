package com.ml4j.optimizer;


/**
 * @author: kexin
 * @date: 2022/6/25 16:09
 **/
public class FixedOptimizer extends Optimizer{
    public FixedOptimizer(float initLearningRate) {
        this.initLearningRate = initLearningRate;
    }

    public FixedOptimizer() {
    }

    @Override
    public float computeLearningRate() {
        return this.initLearningRate;
    }
}
