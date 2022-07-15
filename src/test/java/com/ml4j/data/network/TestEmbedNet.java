package com.ml4j.data.network;

import com.ml4j.data.DenseVector;
import com.ml4j.data.data.DataLoader;
import com.ml4j.data.data.SparseDenseDataSet;
import com.ml4j.initializer.Initializer;
import com.ml4j.initializer.TruncatedNormalInitializer;
import com.ml4j.math.ActivateFunction;
import com.ml4j.math.Identity;
import com.ml4j.math.Sigmoid;
import com.ml4j.network.*;
import com.ml4j.optimizer.FixedOptimizer;
import com.ml4j.optimizer.Optimizer;
import com.ml4j.regularizer.L1Regularizer;
import com.ml4j.regularizer.Regularizer;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.mahout.classifier.evaluation.Auc;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Date: 22-6-27
 * Time: 上午11:33
 */
@Slf4j
public class TestEmbedNet {
    @Test
    public void testEmbedCrossEntropyLoss() throws Exception {
        int sampleNum = 40000;
        SparseDenseDataSet data = DataLoader.loadSparseDenseCensus(sampleNum, 0.3f);
        int vocabSize = data.getSparseFeatNum();
        int embedSize = 5;
        int denseFeatSize = data.getTrainX().get(0).getRight().data().length;

        ActivateFunction sigmoid = new Sigmoid();
        ActivateFunction identity = new Identity();
        List<Layer> layers = new ArrayList<>();
        layers.add(new DenseLayer(10, sigmoid, "first", null));
        layers.add(new DenseLayer(1, identity, "second", null));

        Loss loss = new BinaryLogitWithCrossEntropyLoss();
        Initializer initializer = new TruncatedNormalInitializer();
        Optimizer optimizer = new FixedOptimizer(5e-3f);
        Regularizer regularizer = new L1Regularizer(1e-4f);

        EmbeddingLayer embeddingLayer = new EmbeddingLayer(vocabSize, embedSize, EmbeddingLayer.Combiner.AVG, regularizer);
        DenseLayer firstDenseLayer = new DenseLayer(10, sigmoid);
        firstDenseLayer.setInSize(denseFeatSize);

        NetworkWithEmbedding net = new NetworkWithEmbedding(embeddingLayer, firstDenseLayer, layers, loss, initializer, optimizer);
        net.build();

        int epochNum = 30;
        int iter = 0;

        for (int epoch = 0; epoch < epochNum; epoch++) {
            float epochLoss = 0;
            Auc trainAuc = new Auc();
            for (int i = 0; i < data.getTrainX().size(); i++) {
                Pair<DenseVector, DenseVector> trainX = data.getTrainX().get(i);
                epochLoss += net.train(trainX.getLeft(), trainX.getRight(), data.getTrainY().get(i));

                //pred[i] = net.predict(x[i])[0];
                int label = (int) data.getTrainY().get(i).data()[0];
                float pred = net.predict(trainX.getLeft(), trainX.getRight()).data()[0];
                trainAuc.add(label, pred);
                iter++;
            }

            Auc testAuc = new Auc();
            for (int i = 0; i < data.getTestY().size(); i++) {
                Pair<DenseVector, DenseVector> x = data.getTestX().get(i);
                epochLoss += net.train(x.getLeft(), x.getRight(), data.getTestY().get(i));

                //pred[i] = net.predict(x[i])[0];
                int label = (int) data.getTestY().get(i).data()[0];
                float pred = net.predict(x.getLeft(), x.getRight()).data()[0];
                testAuc.add(label, pred);
                iter++;
            }

            log.info("epoch:{} iter:{} train loss:{} train auc:{} test auc:{}", epoch, iter, epochLoss / sampleNum, trainAuc.auc(), testAuc.auc());
        }
    }
}

