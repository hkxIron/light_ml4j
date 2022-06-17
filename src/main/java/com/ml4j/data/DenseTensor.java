package com.ml4j.data;

import lombok.extern.slf4j.Slf4j;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Date: 22-6-17
 * Time: 下午5:42
 */
@Slf4j
public class DenseTensor extends Tensor {
    private float[] data;

    @Override
    public int[] getShape() {
        return new int[0];
    }
}
