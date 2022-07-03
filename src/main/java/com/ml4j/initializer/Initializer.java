package com.ml4j.initializer;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.Random;

/**
 * @author: kexin
 * @date: 2022/6/25 16:59
 **/
public abstract class Initializer {
    protected static final Random rand = new Random();
    private float mean = 0;
    private float std = 1;

    public Initializer() { }

    public float getMean() {
        return mean;
    }

    public float getStd() {
        return std;
    }

    public Initializer(float mean, float std) {
        this.mean = mean;
        this.std = std;
    }

    public abstract void init(Tensor v);
}