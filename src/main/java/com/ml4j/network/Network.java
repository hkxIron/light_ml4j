package com.ml4j.network;

import com.ml4j.data.DenseVector;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/6/25 13:39
 **/
@Slf4j
@Getter
public class Network {
    private Optimizer optimizer;
    private List<Layer> layers;
    private Loss lossLayer;
    private Initializer initializer;

    public Initializer getInitializer() {
        return initializer;
    }

    public Network(List<Layer> layers, Loss lossLayer, Initializer initializer,
                   Optimizer optimizer) {
        this.layers = layers;
        this.lossLayer = lossLayer;
        this.initializer = initializer;
        this.optimizer = optimizer;
    }

    public void build(int featSize) {
        int layerNum = layers.size();
        log.info("layerNum:{} featSize:{}", layerNum, featSize);
        assert layerNum > 0;
        layers.get(0).setInSize(featSize);
        layers.get(0).initWeights(initializer);

        for (int i = 1; i < layerNum; i++) {
            int outSizeLastLayer = layers.get(i - 1).getOutSize();
            layers.get(i).setInSize(outSizeLastLayer);
            layers.get(i).initWeights(initializer);
        }
    }

    public float forward(DenseVector x, DenseVector y) {
        int num = layers.size();
        assert num > 0;
        DenseVector in = x;
        float loss = 0;
        for (int i = 0; i < num; i++) {
            Layer layer = layers.get(i);
            layer.setInput(in);
            in = layer.forward();
            loss += layer.getRegularizationLoss();
        }
        lossLayer.setInput(in);
        lossLayer.setLabel(y);
        loss += lossLayer.computeLoss();

        return loss;
    }

    public DenseVector backward() {
        int num = layers.size();
        assert num > 0;
        DenseVector delta = lossLayer.computeGrad();
        for (int i = num - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            delta = layer.backward(delta);
        }
        return delta;
    }

    public void update() {
        int num = layers.size();
        assert num > 0;
        for (int i = num - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            layer.update(optimizer);
        }
    }

    public float train(DenseVector x, DenseVector y) {
        float loss = forward(x, y);
        backward();
        update();
        return loss;
    }

    public DenseVector predict(DenseVector x) {
        int num = layers.size();
        assert num > 0;
        DenseVector in = x;
        for (int i = 0; i < num; i++) {
            Layer layer = layers.get(i);
            layer.setInput(in);
            in = layer.forward();
        }
        lossLayer.setInput(in);
        DenseVector score = lossLayer.predict();
        return score;
    }
}
