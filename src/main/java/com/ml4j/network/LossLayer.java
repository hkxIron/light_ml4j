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
public abstract class LossLayer implements Layer{
    private DenseVector input;
    private DenseVector label;

    public abstract float computeLoss(); // batch loss
    public abstract DenseVector predict(); //
    public abstract DenseVector computeGrad();
    public void update(){}
}
