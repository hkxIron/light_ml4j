package com.ml4j.data.network;

import com.ml4j.data.DenseVector;
import com.ml4j.data.data.DataLoader;
import com.ml4j.data.data.SparseDataSet;
import com.ml4j.logisticregression.SparseLogisticRegression;
import lombok.extern.slf4j.Slf4j;
import org.apache.mahout.classifier.evaluation.Auc;
import org.junit.Test;

import java.util.List;
import java.util.Optional;

import static com.ml4j.data.utils.GsonUtil.normalGson;

/**
 * @author: kexin
 * @date: 2022/7/3 10:38
 **/
@Slf4j
public class TestSparseLR {
    @Test
    public void testSparseLR() throws Exception {
        int totalSampleNum = 40000;
        SparseDataSet data = DataLoader.loadCensus(totalSampleNum, 0.3f);
        SparseLogisticRegression model = new SparseLogisticRegression();
        model.setHyperParams(data.getFeatureNames().size(),
                10,
                1e-4f,
                1e-5f,
                1e-5f,
                 Optional.of(data.getFeatureNames())
                );

        model.train(data.getTrainX(), data.getTrainY());
        List<SparseLogisticRegression.Weight> weights = model.getSortedWeight();
        log.info("sorted weights:{}", normalGson.toJson(weights));


        DenseVector trainPred = model.predict(data.getTrainX());
        Auc trainAuc = new Auc();
        for (int i = 0; i < trainPred.data().length; i++) {
            trainAuc.add((int) data.getTrainY().data()[i], trainPred.data()[i]);
        }

        DenseVector pred = model.predict(data.getTestX());
        Auc testAuc = new Auc();
        for (int i = 0; i < pred.data().length; i++) {
            testAuc.add((int) data.getTestY().data()[i], pred.data()[i]);
        }

        log.info("train auc:{} test auc:{}", trainAuc.auc(), testAuc.auc());
    }
}
