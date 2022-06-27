package com.ml4j.data.network;

import com.ml4j.data.DenseVector;
import com.ml4j.data.Initializer;
import com.ml4j.data.NormalInitializer;
import com.ml4j.math.ActivateFunction;
import com.ml4j.math.Sigmoid;
import com.ml4j.network.*;
import com.ml4j.optimizer.FixedOptimizer;
import com.ml4j.optimizer.Optimizer;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;
import org.mortbay.log.Log;

import java.util.ArrayList;
import java.util.List;

import static com.ml4j.data.VectorUtils.maxIndex;
import static com.ml4j.data.utils.FileUtils.getFileAbsolutePath;
import static com.ml4j.data.utils.FileUtils.readFile;
import static com.ml4j.metric.Accurrancy.calculateAcc;

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
        layers.add(new DenseLayer(4, sigmoid, "second"));

        Loss loss = new SoftmaxWithCrossEntropyLoss();
        Initializer initializer = new NormalInitializer();
        Optimizer optimizer = new FixedOptimizer(1e-4f);
        Network net = new Network(layers, loss, initializer, optimizer);
        net.build(inputFeatureDim);

        int sampleNum = 120;
        int epochNum = 100;
        int iter = 0;
        Pair<DenseVector[], DenseVector[]> xy = getIrisData(sampleNum);
        DenseVector[] x = xy.getLeft();
        DenseVector[] y = xy.getRight();

        for (int epoch = 0; epoch < epochNum; epoch++) {
            float trainLoss = 0;
            int[] pred = new int[sampleNum];
            int[] label = new int[sampleNum];
            for (int i = 0; i < sampleNum; i++) {
                trainLoss = net.train(x[i], y[i]);

                pred[i] = maxIndex(net.predict(x[i]));
                label[i] = maxIndex(y[i].data());
                /*
                if (iter % 2000 == 0) {
                    log.info("epoch:{} iter:{} train loss:{}", epoch, iter, trainLoss);
                }
                */
                iter++;
            }
            //log.info("epoch:{} iter:{} train loss:{}", epoch, iter, trainLoss);
            float acc = calculateAcc(pred, label);
            log.info("epoch:{} iter:{} train loss:{} train acc:{}", epoch, iter, trainLoss, acc);
        }
    }

    @Test
    public void testBinaryCrossEntropyLoss() throws Exception {
        ActivateFunction sigmoid = new Sigmoid();
        int inputFeatureDim = 4;

        List<Layer> layers = new ArrayList<>();
        //layers.add(new DenseLayer(5, sigmoid, "first"));
        layers.add(new DenseLayer(4, sigmoid, "second"));

        Loss loss = new BinaryLoigitWithCrossEntropyLoss();
        Initializer initializer = new NormalInitializer();
        Optimizer optimizer = new FixedOptimizer(1e-4f);
        Network net = new Network(layers, loss, initializer, optimizer);
        net.build(inputFeatureDim);

        int sampleNum = 120;
        int epochNum = 100;
        int iter = 0;
        Pair<DenseVector[], DenseVector[]> xy = getIrisData(sampleNum);
        DenseVector[] x = xy.getLeft();
        DenseVector[] y = xy.getRight();
        for (int epoch = 0; epoch < epochNum; epoch++) {
            float trainLoss = 0;
            for (int i = 0; i < sampleNum; i++) {
                trainLoss = net.train(x[i], y[i]);
                if (iter % 2000 == 0) {
                    log.info("epoch:{} iter:{} train loss:{}", epoch, iter, trainLoss);
                }
                iter++;
            }
            //log.info("epoch:{} iter:{} train loss:{}", epoch, iter, trainLoss);
        }
    }
}
