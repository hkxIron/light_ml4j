package com.ml4j.network;

import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.initializer.Initializer;
import com.ml4j.math.ActivateFunction;
import com.ml4j.math.Tanh;
import com.ml4j.optimizer.Optimizer;
import com.ml4j.regularizer.Regularizer;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/6/23 23:10
 * <p>
 * h(t) = tanh(Whx * x(t) + Whh*h(t-1) + bh)
 * y = Why * h(t) + by
 * p = softmax(y)
 * loss = - sum_k{ label_k*log(p_k) }
 **/
public class RNNLayer extends Layer {
    private String name;
    private int inSize;
    private int outSize;
    private int hiddenSize;
    private List<DenseVector> input; // 不定长度的sequence
    private List<DenseVector> hidden; // x0->hidden0, x1->hidden1

    private DenseMatrix Wh; // [hiddenSize, hiddenSize]
    private DenseMatrix Wx; // [hiddenSize, inSize]
    private DenseVector bias; // bias, 输出有多少个节点，就有多少个bias, [1, hiddenSize]

    private DenseVector dWh; // [hiddenSize, hiddenSize]
    private DenseVector dWx; // [hiddenSize, inSize]
    private DenseMatrix dBias;

    private Regularizer regularizer;
    private ActivateFunction function;
    private static final ActivateFunction tanh = new Tanh();

    public void setInput(List<DenseVector> input) {
        this.input = input;
    }

    public RNNLayer(int hiddenSize,
                    ActivateFunction function,
                    String name,
                    Regularizer regularizer) {
        this.hiddenSize = hiddenSize;
        this.function = function;
        this.name = name;
        this.regularizer = regularizer;
    }

    public void setInSize(int size) {
        this.inSize = size;
    }

    @Override
    public void initWeights(Initializer initializer) {
        Wh = new DenseMatrix(new float[hiddenSize][hiddenSize]);
        Wx = new DenseMatrix(new float[hiddenSize][inSize]);
        bias = new DenseVector(new float[hiddenSize]);
        initializer.init(Wh);
        initializer.init(Wx);
        initializer.init(bias);
        /*
        Initializer zero = new ZeroInitializer();
        zero.init(hidden);
        */
    }

    @Override
    public int getOutSize() {
        return this.hiddenSize;
    }

    @Override
    public int getInSize() {
        return this.inSize;
    }

    @Override
    public float getRegularizationLoss() {
        float loss = 0;
        if (this.regularizer != null) {
            loss += regularizer.computeLoss(this.Wh);
            loss += regularizer.computeLoss(this.Wx);
            loss += regularizer.computeLoss(this.bias);
        }
        return loss;
    }

    /**
     * s = W_h*h_(t-1)+ W_x*x_t + bias
     * h_t = tanh(s)
     *
     * @return
     */
    @Override
    public DenseVector forward() {
        int seqLen = input.size();
        DenseVector ht_prev = new DenseVector(new float[hiddenSize]);
        DenseVector ht = null;
        for (int i = 0; i < seqLen; i++) {
            DenseVector xt = input.get(i);
            DenseVector s = (DenseVector) Wh.multiply(ht_prev)
                    .add(Wx.multiply(xt), true).
                            add(bias, true); // [outsize]
            ht = tanh.activate(s, true);
            hidden.add(ht);
            ht_prev = ht;
        }
        return ht;
    }

    /**
     * BPTT: backpropagation through time
     *
     *              Loss
     *            /  |  \
     *           /   |   \
     *         L1    L2   L3 ...
     *         ^     ^     ^
     *         | Wh  | Wh  |
     *   h0 -> h1 -> h2 -> h3 ...
     *         ^     ^     ^
     *         | Wx  | Wx  |
     *         x1    x2    x3 ...
     *
     *        因此可以看到,前向转播时,h1的信息流动: h1->h2->h3
     *        因此求导数时,dh2中应有dh3,dh1中应有dh2
     *
     * forward:
     * s = W_h*h_(t-1)+ W_x*x_t + bias
     * h_t = tanh(s)
     * Loss = sum_t(f(h_t, label_t))
     *
     * backward:
     * dL/ds = dL/dht*dht/ds = delta* tanh'
     *
     * dL/dwh = dL/ds*ds/dWh = dL/ds * h(t-1)
     * dL/dwx = dL/ds*ds/dWx = dL/ds * xt
     * dL/dbias = dL/ds
     *
     *
     * dL/dh(n) = delta
     * dL/dh(n-1) =
     *
     * @param delta
     * @return
     */
    @Override
    public DenseVector backward(DenseVector delta) {
        int seqLen = input.size();
        for (int i = seqLen-1; i >=0; i--) {

        }
        return null;
    }

    /**
     * 在rnn中，每次需要clip梯度
     * for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
     * np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
     *
     * @param optimizer
     */
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
