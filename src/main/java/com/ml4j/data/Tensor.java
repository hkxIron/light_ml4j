package com.ml4j.data;

import java.util.function.BiFunction;
import java.util.function.Function;

import static com.ml4j.initializer.VectorUtils.allEquals;

/**
 * @author: kexin
 * @date: 2022/6/23 21:53
 **/
public abstract class Tensor<T> {
    public abstract int[] getShape();

    public abstract T data();

    public abstract void data(T t);

    public abstract Tensor copy();

    public abstract Tensor valuesLike(float x);

    public abstract Tensor elementWise(Tensor otherTensor, BiFunction<Float, Float, Float> function, boolean inPlace);

    public abstract Tensor elementWise(Function<Float, Float> function, boolean inPlace);

    public abstract float sum();

    public Tensor add(Tensor vec, boolean inPlace) {
        assert allEquals(this.getShape(), vec.getShape());
        return elementWise(vec, (a, b) -> a + b, inPlace);
    }

    public Tensor abs(boolean inPlace) {
        return elementWise(Math::abs, inPlace);
    }

    public Tensor sign(boolean inPlace) {
        return elementWise(Math::signum, inPlace);
    }

    public Tensor minus(Tensor vec, boolean inPlace) {
        assert allEquals(this.getShape(), vec.getShape());
        return elementWise(vec, (a, b) -> a - b, inPlace);
    }

    public Tensor pow(float x, boolean inPlace) {
        return elementWise((a) -> (float) Math.pow(a, x), inPlace);
    }

    public Tensor multiply(float x, boolean inPlace) {
        return elementWise(a -> a * x, inPlace);
    }

    public Tensor multiply(Tensor vec, boolean inPlace) {
        assert allEquals(this.getShape(), vec.getShape());
        return elementWise(vec, (a, b) -> a * b, inPlace);
    }

    public boolean equalsInTolerance(Tensor vec, float eps) {
        return elementWise(vec, (a, b) -> Math.abs(a - b), false)
                .sum() <= eps;
    }
}
