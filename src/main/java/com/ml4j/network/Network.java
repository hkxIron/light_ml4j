package com.ml4j.network;

import com.ml4j.data.DenseVector;
import com.ml4j.data.Initializer;
import com.ml4j.optimizer.Optimizer;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/6/25 13:39
 **/
@Slf4j
public class Network {
    @Getter
    @Setter
    private Optimizer optimizer;
    @Getter
    private List<DenseLayer> mlpLayers;
    @Getter
    private LossLayer lossLayer;
    @Getter
    private Initializer initializer;

    public Network(List<DenseLayer> layers, LossLayer lossLayer, Initializer initializer) {
        this.mlpLayers = layers;
        this.lossLayer = lossLayer;
        this.initializer = initializer;
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
        for (int i = 0; i < num; i++) {
            DenseLayer layer = mlpLayers.get(i);
            layer.setInput(in);
            in = layer.forward();
        }
        lossLayer.setInput(in);
        lossLayer.setLabel(y);
        float loss = lossLayer.computeLoss();
        return loss;
    }

    public void backward() {
        int num = mlpLayers.size();
        assert num > 0;
        DenseVector delta = lossLayer.computeGrad();
        for (int i = num - 1; i >= 0; i--) {
            DenseLayer layer = mlpLayers.get(i);
            delta = layer.backward(delta);
        }
    }

    public void update() {
        int num = mlpLayers.size();
        assert num > 0;
        DenseVector delta = lossLayer.computeGrad();
        for (int i = num - 1; i >= 0; i--) {
            DenseLayer layer = mlpLayers.get(i);
            layer.update();
        }
    }
}
