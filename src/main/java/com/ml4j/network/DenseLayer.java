package com.ml4j.network;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Tensor;
import com.ml4j.initializer.Initializer;
import com.ml4j.math.ActivateFunction;
import com.ml4j.optimizer.Optimizer;
import com.ml4j.regularizer.Regularizer;

/**
 * @author: kexin
 * @date: 2022/6/23 23:10
 **/
public class DenseLayer extends Layer {
    private String name;
    private int inSize;
    private int outSize;
    private DenseVector input;

    private DenseMatrix weight; // [outSize, inSize]
    private DenseVector bias; // 输出有多少个节点，就有多少个bias, [1, outSize]
    private DenseVector wxPlusBias;
    private DenseVector dLdb; // [1* outSize]
    private DenseMatrix dLdW;
    private Regularizer regularizer;
    private ActivateFunction function;

    @Override
    public void setInput(Tensor input) {
        assert input instanceof DenseVector;
        this.input = (DenseVector) input;
    }

    @Override
    public void setOutSize(int size) {
        this.outSize = size;
    }

    public DenseLayer(int outSize, ActivateFunction function) {
        this(outSize, function, "dense", null);
    }

    public DenseLayer(int outSize, ActivateFunction function, String name,
                      Regularizer regularizer) {
        this.outSize = outSize;
        this.function = function;
        this.name = name;
        this.regularizer = regularizer;
    }

    public void setInSize(int size){
       this.inSize = size;
    }

    @Override
    public void initWeights(Initializer initializer) {
        weight = new DenseMatrix(new float[outSize][inSize]);
        bias = new DenseVector(new float[outSize]);
        initializer.init(weight);
        initializer.init(bias);
    }

    @Override
    public int getOutSize() {
        return this.outSize;
    }

    @Override
    public int getInSize() {
        return this.inSize;
    }

    @Override
    public float getRegularizationLoss() {
        float loss = 0;
        if (this.regularizer != null) {
            loss += regularizer.computeLoss(this.weight);
            loss += regularizer.computeLoss(this.bias);
        }
        return loss;
    }

    /**
     * a = Wx + bias
     * P = softmax(a)
     * loss = sum_i(-yi*log(pi))
     *
     * @return
     */
    @Override
    public DenseVector forward() {
        this.wxPlusBias = (DenseVector) weight.multiply(input)
                .add(bias, true); // [outsize]
        DenseVector p = function.activate(wxPlusBias, false);
        return p;
    }

    /**
     * dLoss/dPi = Pi - Yi
     * dLoss/dW =dLoss/dPi * dPi/dai *dai/dW
     * delta = dLoss/dX = dLoss/dPi *dPi/ai * dai/dX
     * <p>
     * delta = dLoss/dPi
     * dPi/dai = f'(ai)
     * <p>
     * dai/dw = d(Wx+b)/dw = x'
     * <p>
     * dai/dX = w'
     */
    @Override
    public DenseVector backward(DenseVector delta) {
        DenseVector dPda = function.gradient(this.wxPlusBias, false);
        DenseVector diff = (DenseVector) delta.multiply(dPda, false); // [1* outSize]

        this.dLdb = (DenseVector) delta.multiply(dPda, false); // [1* outSize]
        this.dLdW = diff.outerProduct(input);

        if (this.regularizer != null) {
            this.dLdW.add(regularizer.computeGrad(this.weight), true);
            this.dLdb.add(regularizer.computeGrad(this.bias), true);
        }

        // weight:[outSize, inSize]
        // diff:[1, outSize]
        // 注意 delta = dL/dX
        DenseVector dLdX = weight.transpose(false).multiply(diff); // [1*inSize]
        return dLdX;
    }

    @Override
    public void update(Optimizer optimizer) {
        // update
        // w = w + (-1)*lr* dL/dW
        // b = b + (-1)*lr* dL/db
        float learningRate = optimizer.computeLearningRate();
        this.weight.add(dLdW.multiply(-learningRate, false), true);
        this.bias.add(dLdb.multiply(-learningRate, false), true);
    }
}
