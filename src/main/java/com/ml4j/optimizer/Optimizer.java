package com.ml4j.optimizer;


import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * @author: kexin
 * @date: 2022/6/25 16:07
 **/
@Getter
@NoArgsConstructor
public abstract class Optimizer {
    protected float initLearningRate = 0;
    protected int globalStep = 0;

    public int addStep(int step) {
        this.globalStep += step;
        return this.globalStep;
    }

    public int addStep() {
        return addStep(1);
    }

    public abstract float computeLearningRate();
}
