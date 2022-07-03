package com.ml4j.network;

import com.ml4j.data.DenseVector;
import com.ml4j.data.Initializer;
import com.ml4j.optimizer.Optimizer;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/6/25 13:39
 **/
@Slf4j
public class Network {
    private Optimizer optimizer;
    private List<Layer> mlpLayers;
    private Loss lossLayer;
    private Initializer initializer;

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public List<Layer> getMlpLayers() {
        return mlpLayers;
    }

    public Loss getLossLayer() {
        return lossLayer;
    }

    public Initializer getInitializer() {
        return initializer;
    }

    public Network(List<Layer> layers, Loss lossLayer, Initializer initializer,
                   Optimizer optimizer) {
        this.mlpLayers = layers;
        this.lossLayer = lossLayer;
        this.initializer = initializer;
        this.optimizer = optimizer;
    }

    public void build(int featSize) {
        int layerNum = mlpLayers.size();
        log.info("layerNum:{} featSize:{}", layerNum, featSize);
        assert layerNum > 0;
        mlpLayers.get(0).initWeights(featSize, initializer);

        for (int i = 1; i < layerNum; i++) {
            int inSize = mlpLayers.get(i - 1).getOutSize();
            mlpLayers.get(i).initWeights(inSize, initializer);
        }
    }

    public float forward(DenseVector x, DenseVector y) {
        int num = mlpLayers.size();
        assert num > 0;
        DenseVector in = x;
        float loss = 0;
        for (int i = 0; i < num; i++) {
            Layer layer = mlpLayers.get(i);
            layer.setInput(in);
            in = layer.forward();
        }
        lossLayer.setInput(in);
        lossLayer.setLabel(y);
        loss += lossLayer.computeLoss();

        return loss;
    }

    public void backward() {
        int num = mlpLayers.size();
        assert num > 0;
        DenseVector delta = lossLayer.computeGrad();
        for (int i = num - 1; i >= 0; i--) {
            Layer layer = mlpLayers.get(i);
            delta = layer.backward(delta);
        }
    }

    public void update() {
        int num = mlpLayers.size();
        assert num > 0;
        for (int i = num - 1; i >= 0; i--) {
            Layer layer = mlpLayers.get(i);
            layer.update(optimizer);
        }
    }

    public float train(DenseVector x, DenseVector y) {
        float loss = forward(x, y);
        backward();
        update();
        return loss;
    }

    public float[] predict(DenseVector x) {
        int num = mlpLayers.size();
        assert num > 0;
        DenseVector in = x;
        for (int i = 0; i < num; i++) {
            Layer layer = mlpLayers.get(i);
            layer.setInput(in);
            in = layer.forward();
        }
        lossLayer.setInput(in);
        DenseVector score = lossLayer.predict();
        return score.data();
    }
}
