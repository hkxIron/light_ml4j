package com.ml4j.data;


import lombok.Getter;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;

/**
 * @author: kexin
 * @date: 2022/6/29 10:47
 **/
public class SparseVector extends Tensor<Map<Integer, Float>> {
    @Getter
    private final Map<Integer, Float> indToVal; //  0,  7,  9
    private int maxSize;

    public int getMaxSize() {
        return maxSize;
    }

    public SparseVector(int maxSize) {
        this.indToVal = new HashMap<>(maxSize);
        this.maxSize = maxSize;
    }

    public SparseVector(int maxSize, Map<Integer, Float> indToVal) {
        assert indToVal.size() <= maxSize;
        this.maxSize = maxSize;
        indToVal.keySet().forEach(this::checkIndex);
        this.indToVal = indToVal;
    }

    @Override
    public int[] getShape() {
        return new int[]{maxSize};
    }

    @Override
    public Map<Integer, Float> data() {
        return indToVal;
    }

    @Override
    public void data(Map<Integer, Float> integerFloatMap) {
        throw new RuntimeException("not implement method");
    }

    @Override
    public Tensor copy() {
        SparseVector vec = new SparseVector(maxSize);
        vec.indToVal.putAll(this.indToVal);
        return vec;
    }

    @Override
    public Tensor valuesLike(float x) {
        SparseVector vec = new SparseVector(maxSize);
        this.indToVal.forEach((k, v) -> {
            vec.indToVal.put(k, x);
        });
        return vec;
    }

    @Override
    public boolean equalsInTolerance(Tensor vec0, float eps) {
        if (!(vec0 instanceof SparseVector)) {
            return false;
        }
        SparseVector vec = (SparseVector) vec0;
        if (vec.maxSize != maxSize || vec.indToVal.size() != vec.indToVal.size()) {
            return false;
        }
        return elementWise(vec, (a, b) -> Math.abs(a - b), false)
                .sum() <= Math.abs(eps);
    }

    public float get(int i) {
        checkIndex(i);
        return indToVal.getOrDefault(i, 0f);
    }

    public void set(int i, float val) {
        checkIndex(i);
        indToVal.put(i, val);
    }

    public void checkIndex(int i) {
        assert i >= 0 && i < maxSize;
    }

    public static <T> List<T> union(final Set<T> s1, final Set<T> s2) {
        Set<T> s = new TreeSet<>(s1); // 具有内部排序
        s.addAll(s2);
        return new ArrayList<>(s);
    }

    public float innerProduct(SparseVector vec) {
        assert maxSize == vec.maxSize;
        double res = 0;
        for (Integer ind : union(this.indToVal.keySet(), vec.indToVal.keySet())) {
            res += this.get(ind) * vec.get(ind);
        }
        return (float) res;
    }

    public SparseVector gather(int[] inds) {
        Map<Integer, Float> weight = new HashMap<>(inds.length);
        for (int ind : inds) {
            checkIndex(ind);
            weight.put(ind, this.get(ind));
        }
        return new SparseVector(inds.length, weight);
    }


    public float reduce(float initValue, final BinaryOperator<Float> function) {
        return this.indToVal.values().stream().reduce(initValue, function);
    }

    public float sum() {
        return this.reduce(0, (a, b) -> a + b);
    }

    /**
     * 计算所有元素的乘积
     *
     * @return
     */
    public float product() {
        return this.reduce(1, (a, b) -> a * b);
    }

    @Override
    public SparseVector elementWise(final Function<Float, Float> function, boolean inPlace) {
        Map<Integer, Float> weight;
        if (inPlace) {
            weight = this.indToVal;
        } else {
            weight = new HashMap<>(this.indToVal);
        }
        weight.forEach((ind, val) -> {
            weight.put(ind, function.apply(val));
        });
        if (inPlace) {
            return this;
        } else {
            return new SparseVector(this.maxSize, weight);
        }
    }

    @Override
    public SparseVector elementWise(final Tensor vec0, final BiFunction<Float, Float, Float> function, boolean inPlace) {
        SparseVector vec = (SparseVector) vec0;

        assert this.maxSize == vec.maxSize;
        List<Integer> keys = union(this.indToVal.keySet(), vec.indToVal.keySet());
        Map<Integer, Float> weight;
        if (inPlace) {
            weight = this.indToVal;
        } else {
            weight = new HashMap<>(keys.size());
        }
        for (Integer ind : keys) {
            weight.put(ind, function.apply(this.get(ind), vec.get(ind)));
        }
        if (inPlace) {
            return this;
        } else {
            return new SparseVector(this.maxSize, weight);
        }
    }

    /*
    public SparseVector multiply(float x, boolean inPlace) {
        return elementWise((a) -> a * x, inPlace);
    }

    public SparseVector pow(float x, boolean inPlace) {
        return elementWise((a) -> (float) Math.pow(a, x), inPlace);
    }

    public SparseVector add(float x, boolean inPlace) {
        return elementWise((a) -> a + x, inPlace);
    }

    public SparseVector sign(boolean inPlace) {
        return elementWise(Math::signum, inPlace);
    }

    public SparseVector abs(boolean inPlace) {
        return elementWise(Math::abs, inPlace);
    }
    */

    /*
    public SparseVector multiply(SparseVector x) {
        return elementWise(x, (a, b) -> a * b);
    }

    public SparseVector add(SparseVector x) {
        return elementWise(x, (a, b) -> a + b);
    }

    public SparseVector minus(SparseVector x) {
        return elementWise(x, (a, b) -> a - b);
    }
    */
}
