package com.ml4j.network;

import com.ml4j.data.DenseVector;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * @author: kexin
 * @date: 2022/6/25 15:56
 **/
@NoArgsConstructor
@Setter
@Getter
public abstract class Loss {
    private DenseVector input;
    private DenseVector label;

    protected DenseVector pred;
    //protected float loss;
    //protected DenseVector delta; // 即为dL/dX

    public abstract float computeLoss(); // batch loss

    public abstract DenseVector predict(); //

    public abstract DenseVector computeGrad();

    /*
    public void update() {
    }

    public DenseVector forward() {
        return null;
    }

    public DenseVector backward(DenseVector diff) {
        return null;
    }
    */
}
