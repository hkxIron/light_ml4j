package com.ml4j.optimizer;

/**
 * @author: kexin
 * @date: 2022/6/25 16:07
 **/
public abstract class Optimizer {
    protected float initLearningRate = 0;
    protected int globalStep = 0;

    public Optimizer() {
    }

    public float getInitLearningRate() {
        return initLearningRate;
    }

    public int getGlobalStep() {
        return globalStep;
    }

    public int addStep(int step) {
        this.globalStep += step;
        return this.globalStep;
    }

    public int addStep() {
        return addStep(1);
    }

    public abstract float computeLearningRate();
}
