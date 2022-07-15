package com.ml4j.network;

import com.ml4j.data.*;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;
import com.ml4j.regularizer.Regularizer;
import lombok.Getter;
import lombok.Setter;

/**
 * @author: kexin
 * @date: 2022/7/3 17:44
 **/
public class EmbeddingLayer extends Layer {
    public enum Combiner {
        AVG,
        SUM
    }

    @Getter
    @Setter
    private DenseMatrix weight; // [vocab_size, embedding_size]

    @Getter
    private int inSize;

    @Getter
    private int outSize;
    private DenseVector input;
    private Combiner combiner;
    private DenseVector dLdW;
    private Regularizer regularizer;

    public EmbeddingLayer(int vocabSize, int embeddingSize, Combiner combiner, Regularizer regularizer) {
        this.inSize = vocabSize;
        this.outSize = embeddingSize;
        this.combiner = combiner;
        this.regularizer = regularizer;
    }

    @Override
    public DenseVector forward() {
        float[] ids = this.input.data();
        DenseVector vec = new DenseVector(outSize);
        for (int i = 0; i < ids.length; i++) {
            int idx = (int) ids[i];
            vec.add(new DenseVector(weight.data()[idx]), true);
        }
        if (combiner == Combiner.AVG) {
            vec.multiply(1f / ids.length, true);
        }
        return vec;
    }

    /**
     * X(i+1) = embedding_layer(weight, ids, SUM)
     * delta = dLoss/dX(i+1)
     * <p>
     * SUM:
     * X(i+1) = X(i)1+X(i)2+...X(i)n
     * <p>
     * AVG:
     * X(i+1) = [X(i)1+X(i)2+...X(i)n]/n
     *
     * @param delta
     * @return
     */
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
            int i = (int) ((float)e);
            if (i >= 0 && i < inSize) { // check index range
                return 0f; // 合法
            } else {
                return 1f;
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
