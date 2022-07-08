package com.ml4j.network;

import com.ml4j.data.DenseVector;
import com.ml4j.data.SparseVector;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;
import jdk.nashorn.internal.objects.annotations.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/6/25 13:39
 **/
@Slf4j
public class NetworkWithEmbedding extends Network {
    private EmbeddingLayer firstEmbedLayer;
    private DenseLayer firstDenseLayer;
    private ConcatLayer concatLayer;

    public NetworkWithEmbedding(EmbeddingLayer firstEmbeddingLayer,
                                DenseLayer firstDenseLayer,
                                List<Layer> middleLayers,
                                Loss lossLayer,
                                Initializer initializer,
                                Optimizer optimizer) {
        super(middleLayers, lossLayer, initializer, optimizer);
        this.firstEmbedLayer = firstEmbeddingLayer;
        this.firstDenseLayer = firstDenseLayer;
        this.concatLayer = new ConcatLayer(this.firstEmbedLayer, this.firstDenseLayer);
    }

    public void build() {
        int outSize = firstEmbedLayer.getOutSize()+ firstDenseLayer.getOutSize();
        concatLayer.initWeights(this.getInitializer());
        super.build(outSize);
    }

    public float forward(DenseVector sparseFeats, DenseVector denseFeats, DenseVector y) {
        this.firstEmbedLayer.setInput(sparseFeats);
        this.firstDenseLayer.setInput(denseFeats);
        DenseVector out = concatLayer.forward();
        float loss = super.forward(out, y);
        return loss;
    }

    @Override
    public DenseVector backward() {
        DenseVector delta = super.backward();
        return this.concatLayer.backward(delta);
    }

    @Override
    public void update() {
       super.update();
       concatLayer.update(this.getOptimizer());
    }

    public float train(DenseVector sparseFeats, DenseVector denseVector, DenseVector y) {
        float loss = this.forward(sparseFeats, denseVector, y);
        this.backward();
        this.update();
        return loss;
    }

    public DenseVector predict(DenseVector sparseFeats, DenseVector denseFeats) {
        this.firstEmbedLayer.setInput(sparseFeats);
        this.firstDenseLayer.setInput(denseFeats);
        DenseVector out = concatLayer.forward();
        return super.predict(out);
    }
}

