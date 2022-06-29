package com.ml4j.data;

import java.util.HashMap;
import java.util.Map;

/**
 * @author: kexin
 * @date: 2022/6/29 10:47
 **/
public class SparseVector implements SparseTensor {
    private final Map<Integer, Float> indToVal; //  0,  7,  9
    private int size;

    public SparseVector(int size) {
        this.indToVal = new HashMap<>(size);
        this.size = size;
    }

    @Override
    public int[] getShape() {
        return new int[]{size};
    }

    public float get(int i) {
        assert i >= 0 && i < size;
        return indToVal.getOrDefault(i, 0f);
    }

    public void set(int i, float val) {
        assert i > 0 && i < size;
        indToVal.put(i, val);
    }
}
