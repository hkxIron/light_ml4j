package com.ml4j.network;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import lombok.Getter;

/**
 * @author: kexin
 * @date: 2022/6/23 22:57
 **/
public interface Layer {
    DenseVector forward();
    DenseVector backward(DenseVector delta);
    void update();
}
