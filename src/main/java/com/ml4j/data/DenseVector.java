package com.ml4j.data;

import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * @author: kexin
 * @date: 2022/6/23 21:44
 **/
@NoArgsConstructor
public class DenseVector implements Tensor<float[]> {
    private float[] data;

    public DenseVector(float[] data) {
        this.data = data;
    }

    @Override
    public int[] getShape() {
        return new int[]{data.length};
    }

    @Override
    public float[] data() {
        return this.data;
    }

    @Override
    public void data(float[] floats) {
        this.data = floats;
    }
}
