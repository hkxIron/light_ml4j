package com.ml4j.data;

/**
 * @author: kexin
 * @date: 2022/6/23 21:53
 **/
public interface Tensor<T> {
    int[] getShape();
    T data();
    void data(T t);
}
