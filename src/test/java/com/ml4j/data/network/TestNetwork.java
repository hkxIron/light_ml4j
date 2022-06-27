package com.ml4j.data.network;

import com.google.common.collect.Lists;
import com.ml4j.data.DenseVector;
import com.ml4j.data.Initializer;
import com.ml4j.data.NormalInitializer;
import com.ml4j.data.utils.GsonUtil;
import com.ml4j.math.ActivateFunction;
import com.ml4j.math.Identity;
import com.ml4j.math.Sigmoid;
import com.ml4j.network.*;
import com.ml4j.optimizer.FixedOptimizer;
import com.ml4j.optimizer.Optimizer;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.mahout.classifier.evaluation.Auc;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static com.ml4j.data.VectorUtils.maxIndex;
import static com.ml4j.data.utils.FileUtils.getFileAbsolutePath;
import static com.ml4j.data.utils.FileUtils.readFile;
import static com.ml4j.metric.Accuracy.calculateAcc;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Email: hukexin@xiaomi.com
 * Date: 22-6-27
 * Time: 上午11:33
 */
@Slf4j
public class TestNetwork {
    public List<Pair<DenseVector[], DenseVector[]>> split(Pair<DenseVector[], DenseVector[]> xy, float trainRatio) {
        int all = xy.getLeft().length;
        int trainSize = (int) (all * trainRatio);
        DenseVector x = xy.getLeft();
        DenseVector y = xy.getRight();

        DenseVector[] trainX = new DenseVector[trainSize];
        DenseVector[] trainY = new DenseVector[trainSize];

        for (int i = 0; i < trainSize; i++) {
            trainX[i] = x[i];
            trainY[i] = y[i];
        }

        DenseVector[] testX = new DenseVector[all - trainSize];
        DenseVector[] testY = new DenseVector[all - trainSize];
        for (int i = trainSize; i < all; i++) {
            testX[i] = x[i];
            testY[i] = y[i];
        }

        return Lists.newArrayList(Pair.of(trainX, trainY), Pair.of(testX, testY));
    }

    public Pair<DenseVector[], DenseVector[]> getIrisData(int sampleNum) throws Exception {
        String content = readFile(getFileAbsolutePath("iris.csv"));
        DenseVector[] xs = new DenseVector[sampleNum];
        DenseVector[] ys = new DenseVector[sampleNum];

        String[] lines = content.split("\n");
        int j = 0;
        //int pos = 0;
        for (int i = 0; i < lines.length; i++) {
            if (i == 0) {
                continue;
            }
            String[] arr = lines[i].split(",");
            float[] x = new float[4];
            x[0] = Float.valueOf(arr[0]);
            x[1] = Float.valueOf(arr[1]);
            x[2] = Float.valueOf(arr[2]);
            x[3] = Float.valueOf(arr[3]);

            float[] y = new float[4];
            int index = Integer.valueOf(arr[4]);
            y[index] = 1;

            if (j == sampleNum) {
                break;
            }
            xs[j] = new DenseVector(x);
            ys[j] = new DenseVector(y);
            j++;
        }

        //System.out.println("positive:" + pos + " negative:" + (sampleNum - pos));
        return Pair.of(xs, ys);
    }

    @Test
    public void testSoftmaxWithCrossEntropyLoss() throws Exception {
        ActivateFunction sigmoid = new Sigmoid();
        int inputFeatureDim = 4;

        List<Layer> layers = new ArrayList<>();
        //layers.add(new DenseLayer(5, sigmoid, "first"));
        layers.add(new DenseLayer(5, sigmoid, "second"));
        layers.add(new DenseLayer(4, new Identity(), "second"));

        Loss loss = new SoftmaxWithCrossEntropyLoss();
        Initializer initializer = new NormalInitializer();
        Optimizer optimizer = new FixedOptimizer(5e-3f);
        Network net = new Network(layers, loss, initializer, optimizer);
        net.build(inputFeatureDim);

        int sampleNum = 120;
        int epochNum = 100;
        int iter = 0;
        Pair<DenseVector[], DenseVector[]> xy = getIrisData(sampleNum);
        DenseVector[] x = xy.getLeft();
        DenseVector[] y = xy.getRight();

        int[] pred = new int[sampleNum];
        int[] label = new int[sampleNum];

        for (int epoch = 0; epoch < epochNum; epoch++) {
            float epochLoss = 0;
            for (int i = 0; i < sampleNum; i++) {
                epochLoss += net.train(x[i], y[i]);

                pred[i] = maxIndex(net.predict(x[i]));
                label[i] = maxIndex(y[i].data());
                iter++;
            }
            float acc = calculateAcc(pred, label);
            log.info("epoch:{} iter:{} train loss:{} train acc:{}", epoch, iter, epochLoss / sampleNum, acc);
        }
        log.info("label:{}", GsonUtil.normalGson.toJson(label));
        log.info("pred:{}", GsonUtil.normalGson.toJson(pred));
    }

    @Test
    public void testBinaryCrossEntropyLoss() throws Exception {
        ActivateFunction sigmoid = new Sigmoid();
        ActivateFunction identity = new Identity();
        int inputFeatureDim = 4;

        List<Layer> layers = new ArrayList<>();
        layers.add(new DenseLayer(5, sigmoid, "first"));
        layers.add(new DenseLayer(1, identity, "second"));

        Loss loss = new BinaryLogitWithCrossEntropyLoss();
        Initializer initializer = new NormalInitializer();
        Optimizer optimizer = new FixedOptimizer(5e-3f);
        Network net = new Network(layers, loss, initializer, optimizer);
        net.build(inputFeatureDim);

        int sampleNum = 120;
        int epochNum = 40;
        int iter = 0;
        List<Pair<DenseVector[], DenseVector[]>> xy = split(getIrisData(sampleNum), 0.7);
        DenseVector[] x = xy.get(0).getLeft();
        DenseVector[] oneHotY = xy(0).getRight();

        DenseVector[] testX = xy.get(1).getLeft();
        DenseVector[] testOneHotY = xy(1).getRight();

        DenseVector[] binaryY = new DenseVector[oneHotY.length];
        int[] originLabel = new int[sampleNum];
        int posNum = 0;
        for (int i = 0; i < oneHotY.length; i++) {
            int pos = oneHotY[i].data()[0] == 1 ? 1 : 0;
            posNum += pos;
            originLabel[i] = pos;
            binaryY[i] = new DenseVector(new float[]{pos});
        }

        System.out.println("posNUm:" + posNum + " negNum:" + (sampleNum - posNum) + " sampleNum:" + sampleNum);
        float[] predScore = new float[sampleNum];

        for (int epoch = 0; epoch < epochNum; epoch++) {
            float epochLoss = 0;
            Auc auc = new Auc();
            for (int i = 0; i < sampleNum; i++) {
                epochLoss += net.train(x[i], binaryY[i]);
                //pred[i] = net.predict(x[i])[0];
                int label = (int) binaryY[i].data()[0];
                float pred = net.predict(x[i])[0];
                predScore[i] = pred;
                auc.add(label, pred);
                iter++;
            }
            log.info("epoch:{} iter:{} train loss:{} train auc:{}", epoch, iter, epochLoss / sampleNum, auc.auc());
        }
        log.info("pred :{}", GsonUtil.normalGson.toJson(predScore));
        log.info("label:{}", GsonUtil.normalGson.toJson(originLabel));
    }
}
