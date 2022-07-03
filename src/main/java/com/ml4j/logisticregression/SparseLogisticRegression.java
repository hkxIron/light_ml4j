package com.ml4j.logisticregression;

import com.ml4j.data.DenseVector;
import com.ml4j.data.SparseVector;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static com.ml4j.math.FunctionUtils.sigmoid;

/**
 * @author: kexin
 * @date: 2022/7/2 16:46
 **/
@NoArgsConstructor
@Slf4j
public class SparseLogisticRegression {
    public static class Weight{
        public String name;
        public float value;
        public int index;

        public Weight(String name, int index, float value) {
            this.name = name;
            this.index = index;
            this.value = value;
        }
    }

    private int featNum;
    private int trainEpoch;
    private float learningRate; // 学习率
    private float l1Penalty;
    private float l2Penalty;
    @Getter
    private SparseVector weights;
    @Getter
    private float bias;
    private Optional<List<String>> featNames;

    public void setHyperParams(int featNum,
                               int trainEpochNum,
                               float learningRate,
                               float l1Penalty,
                               float l2Penalty,
                               Optional<List<String>> featNames) {
        this.featNum = featNum;
        this.trainEpoch = trainEpochNum;
        this.learningRate = learningRate;
        this.l1Penalty = l1Penalty;
        this.l2Penalty = l2Penalty;
        this.weights = new SparseVector(featNum);
        this.bias = 0;
        featNames.ifPresent(names -> {
            assert featNum == names.size();
        });
        this.featNames = featNames;
    }

    public float train(List<SparseVector> xs, DenseVector ys) {
        assert xs.size() == ys.data().length;
        assert xs.get(0).getMaxSize() == this.featNum;

        int sampleNum = ys.data().length;
        float loss = 0;
        int iter = 0;

        // prob = sigmoid(w*x+bias)
        // -y*log(Prob)-(1-y)*log(1-Prob)
        for (int epoch = 0; epoch < this.trainEpoch; epoch++) {
            loss = 0;
            for (int i = 0; i < sampleNum; i++) {
                //forward
                float prob = this.predict(xs.get(i));
                // loss
                loss += getLoss(prob, ys.data()[i]);
                // compute diff = dL/dx, (pi-yi)*xi
                float diff = prob - ys.data()[i];
                float gradBias = diff;
                SparseVector gradWeight = (SparseVector) xs.get(i).multiply(diff, false);

                /*  loss = |w|
                    d|w|= 1 if w>0
                      -1 if w<0
                      0 if w=0
                */
                if (l1Penalty > 0) { // 只计算sparse处的L1梯度
                    loss += l1Penalty * (weights.abs(false).sum() + Math.abs(bias));

                    SparseVector gradL1 = (SparseVector) weights.sign(false).multiply(l1Penalty, true);

                    gradWeight.add(gradL1, true);
                    gradBias += l1Penalty * Math.signum(bias);
                }
                // loss = w^2
                // d(w^2) = 2*w, 2可以去掉，即为w
                if (l2Penalty > 0) { // 只计算sparse处的L2梯度
                    loss += l2Penalty * (weights.pow(2f, false).sum() + bias * bias);

                    SparseVector gradL2 = (SparseVector) weights.multiply(2 * l2Penalty, false);
                    gradWeight.add(gradL2, true);
                    gradBias += l2Penalty * bias;
                }

                // backward
                // w += -lr* grad;
                weights = (SparseVector) weights.add(gradWeight.multiply(-learningRate, false), false);
                bias += -learningRate * gradBias;

                iter++;
            }
            loss /=sampleNum;
            log.info("epoch:{} iter:{} loss:{}", epoch, iter, loss);
        }
        return loss;
    }

    public DenseVector predict(List<SparseVector> xs) {
        int sampleNum = xs.size();
        float[] prob = new float[sampleNum];
        for (int i = 0; i < xs.size(); i++) {
            prob[i] = predict(xs.get(i));
        }
        return new DenseVector(prob);
    }

    public float predict(SparseVector xs) {
        return sigmoid(xs.innerProduct(weights) + bias);
    }

    public List<Weight> getSortedWeight(){
        assert featNames.isPresent();
        List<Weight> weightList = new ArrayList<>();
        this.weights.getIndToVal().forEach((ind, val)->{
            String name = featNames.get().get(ind);
            weightList.add(new Weight(name, ind, val));
        });
        weightList.sort((o1, o2) -> - Float.compare (o1.value , o2.value));
        return weightList;
    }

    // -y*log(Prob)-(1-y)*log(1-Prob)
    public float getLoss(float score, float y) {
        //assert score > 0 && score < 1;
        double loss;
        if ((int) y == 1) {
            loss = -Math.log(score);
        } else {
            loss = -Math.log(1 - score);
        }
        return (float) loss;
    }

    public float getLoss(float[] scores, float[] ys) {
        assert scores.length == ys.length;
        double loss = 0;
        for (int i = 0; i < scores.length; i++) {
            loss += getLoss(scores, ys);
        }
        return (float) loss;
    }
}
