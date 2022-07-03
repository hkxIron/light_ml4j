package com.ml4j.network;

import com.ml4j.data.DenseVector;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * @author: kexin
 * @date: 2022/6/25 15:56
 **/
public abstract class Loss {
    private DenseVector input;
    private DenseVector label;
    protected DenseVector pred;
    //protected DenseVector delta; // 即为dL/dX

    public void setInput(DenseVector input) {
        this.input = input;
    }

    public void setLabel(DenseVector label) {
        this.label = label;
    }

    public void setPred(DenseVector pred) {
        this.pred = pred;
    }


    public DenseVector getInput() {
        return input;
    }

    public DenseVector getLabel() {
        return label;
    }

    public DenseVector getPred() {
        return pred;
    }


    public Loss(DenseVector input, DenseVector label, DenseVector pred) {
        this.input = input;
        this.label = label;
        this.pred = pred;
    }


    public abstract float computeLoss(); // batch loss
    public abstract DenseVector computeGrad();
    public abstract DenseVector predict(); //
    public Loss() {
    }
}
