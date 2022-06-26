package com.ml4j.optimizer;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * @author: kexin
 * @date: 2022/6/25 16:09
 **/
@NoArgsConstructor
public class FixedOptimizer extends Optimizer{
    public FixedOptimizer(float initLearningRate) {
        this.initLearningRate = initLearningRate;
    }

    @Override
    public float computeLearningRate() {
        return this.initLearningRate;
    }
}
