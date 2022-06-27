package com.ml4j.network;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Initializer;
import com.ml4j.math.ActivateFunction;
import com.ml4j.optimizer.Optimizer;
import lombok.Getter;
import lombok.Setter;

/**
 * @author: kexin
 * @date: 2022/6/23 23:10
 **/
public class DenseLayer implements Layer {
    @Getter
    private String name;
    @Setter
    private DenseVector input; // [1, inSize]

    private int outSize;

    @Getter
    private int inSize;
    private ActivateFunction function;

    private DenseMatrix weight; // [outSize, inSize]
    private DenseVector bias; // 输出有多少个节点，就有多少个bias, [1, outSize]
    private DenseVector wxPlusBias;
    private DenseVector dLdb; // [1* outSize]
    private DenseMatrix dLdW;

    public DenseLayer(int outSize, ActivateFunction function) {
        this(outSize, function, "dense");
    }

    public DenseLayer(int outSize, ActivateFunction function, String name) {
        this.outSize = outSize;
        this.function = function;
        this.name = name;
    }

    @Override
    public void initWeights(int inSize, Initializer initializer) {
        this.inSize = inSize;
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

    /**
     * a = Wx + bias
     * P = softmax(a)
     * loss = sum_i(-yi*log(pi))
     *
     * @return
     */
    @Override
    public DenseVector forward() {
        this.wxPlusBias = weight.multiply(input)
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
        DenseVector diff = delta.elementWiseMultiply(dPda, false); // [1* outSize]

        this.dLdb = delta.elementWiseMultiply(dPda, false); // [1* outSize]
        this.dLdW = diff.outerProduct(input);
        // TODO:L2 regularization

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
        float learingRate = optimizer.getInitLearningRate();
        this.weight.add(dLdW.multiply(-learingRate, false), true);
        this.bias.add(dLdb.multiply(-learingRate, false), true);
    }
}
