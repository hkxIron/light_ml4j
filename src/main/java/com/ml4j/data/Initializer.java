package com.ml4j.data;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.Random;

/**
 * @author: kexin
 * @date: 2022/6/25 16:59
 **/
@Getter
@NoArgsConstructor
public abstract class Initializer {
    protected static final Random rand = new Random();
    private float mean = 0;
    private float std = 1;

    public Initializer(float mean, float std) {
        this.mean = mean;
        this.std = std;
    }

    public abstract void init(DenseVector v);

    public abstract void init(DenseMatrix x);
}