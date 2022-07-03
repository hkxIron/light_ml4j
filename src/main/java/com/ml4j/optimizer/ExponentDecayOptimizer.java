package com.ml4j.optimizer;

import lombok.NoArgsConstructor;

/**
 * @author: kexin
 * @date: 2022/6/25 16:09
 **/
public class ExponentDecayOptimizer extends Optimizer {
    private float decayRate; // 0.99
    private int decaySteps; // 10000
    private float lr;
    private int lastStepStage; //global_step/decay_steps

    public ExponentDecayOptimizer() {
    }

    public ExponentDecayOptimizer(float initLearningRate, float decayRate, int decaySteps) {
        this.initLearningRate = initLearningRate;
        this.decayRate = decayRate;
        this.decaySteps = decaySteps;
        this.lr = initLearningRate;
        this.lastStepStage = 0;
    }

    /**
     * lr = initLearningRate * decayRate^ int(global_step/decay_steps)
     * s     * @return
     */
    @Override
    public float computeLearningRate() {
        this.addStep();
        int stage = globalStep / decaySteps;
        if (stage > lastStepStage) {
            lr *= decayRate;
            lastStepStage = stage;
        }
        return lr;
    }
}
