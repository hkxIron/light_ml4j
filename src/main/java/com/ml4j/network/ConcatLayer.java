package com.ml4j.network;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

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

    public ConcatLayer(Layer left, Layer right) {
        this.left = left;
        this.right = right;

        this.inSize = left.getOutSize() + right.getOutSize();
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
        List<Integer> sizes = new ArrayList<>();
        sizes.add(left.getOutSize());
        sizes.add(right.getOutSize());

        List<DenseVector> dLossDx = delta.split(sizes);
        assert dLossDx.size() == 2;
        left.backward(dLossDx.get(0));
        left.backward(dLossDx.get(1));

        return null;
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
    public void update(Optimizer optimizer){
        this.left.update(optimizer);
        this.right.update(optimizer);
    }

    @Override
    public void initWeights(Initializer initializer) {
        this.left.initWeights(initializer);
        this.right.initWeights(initializer);
    }

    @Override
    public float getRegularizationLoss(){
        return left.getRegularizationLoss()+right.getRegularizationLoss();
    }
}
