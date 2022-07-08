package com.ml4j.network;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;
import com.ml4j.regularizer.Regularizer;
import lombok.Getter;
import lombok.Setter;

/**
 * @author: kexin
 * @date: 2022/7/3 17:44
 **/
public class ConcatLayer extends Layer {
    private Layer left; //
    private Layer right; //

    @Getter
    private int inSize;

    @Getter
    private int outSize;
    private DenseVector dLdW;

    public ConcatLayer(Layer left, Layer right) {
        this.left = left;
        this.right = right;

        this.inSize = left.getOutSize()+ right.getOutSize();
        this.outSize = left.getOutSize() + right.getOutSize();
    }

    @Override
    public DenseVector forward() {
        DenseVector leftOut = left.forward();
        DenseVector rightOut = right.forward();
        return leftOut.concat(rightOut);
    }

    @Override
    public DenseVector backward(DenseVector delta) {
        dLdW = delta.copy();

        int size = input.getShape()[0];
        if (combiner == Combiner.AVG) {
            dLdW.multiply(1f / size, true);
        }

        if (this.regularizer != null) {
            DenseVector regGrad = new DenseVector(outSize);
            float[] ids = this.input.data();
            for (int i = 0; i < ids.length; i++) {
                int idx = (int) ids[i];
                DenseVector row = new DenseVector(weight.data()[idx]);
                regGrad.add(regularizer.computeGrad(row), true);
            }
            if(combiner == Combiner.AVG){
                regGrad.multiply(1f / size, true);
            }
            dLdW.add(regGrad, true);
        }
        return dLdW;
    }

    @Override
    public void update(Optimizer optimizer) {
        // update
        // w = w + (-1)*lr* dL/dW
        float learningRate = optimizer.computeLearningRate();
        DenseVector diff = (DenseVector) dLdW.multiply(-learningRate, false);

        float[] ids = this.input.data();
        for (int i = 0; i < ids.length; i++) {
            int idx = (int) ids[i];
            DenseVector row = new DenseVector(weight.data()[idx]);
            row.add(diff, true);
        }
    }

    @Override
    public void initWeights(Initializer initializer) {
        this.weight = new DenseMatrix(this.inSize, outSize);
        initializer.init(weight);
    }

    @Override
    public void setInput(Tensor x) {
        assert x instanceof DenseVector;
        assert x.elementWise(e -> {
            int i = (int) e;
            if (i >= 0 && i < inSize) { // check index range
                return 0; // 合法
            } else {
                return 1;
            }
        }, false).sum() == 0;
        this.input = (DenseVector) x;
    }

    @Override
    public int getOutSize() {
        return outSize;
    }

    @Override
    public void setOutSize(int size) {
        this.outSize = size;
    }

    @Override
    public void setInSize(int size) {
        this.inSize = size;
    }

    @Override
    public int getInSize() {
        return inSize;
    }

    @Override
    public float getRegularizationLoss() {
        float loss = 0;
        if (this.regularizer != null) {
            float[] ids = this.input.data();
            for (int i = 0; i < ids.length; i++) {
                int idx = (int) ids[i];
                DenseVector row = new DenseVector(weight.data()[idx]);
                loss += regularizer.computeLoss(row);
            }
            if(combiner == Combiner.AVG){
                loss/=ids.length;
            }
        }
        return loss;
    }
}
