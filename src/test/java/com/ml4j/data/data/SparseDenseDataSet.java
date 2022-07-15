package com.ml4j.data.data;

import com.ml4j.data.DenseVector;
import com.ml4j.data.SparseVector;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.tuple.Pair;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/7/2 22:49
 **/
@Getter
@Setter
public class SparseDenseDataSet {
    private List<String> featureNames;
    private int sparseFeatNum;
    private List<Pair<DenseVector, DenseVector>> trainX;
    private List<DenseVector> trainY;

    private List<Pair<DenseVector, DenseVector>> testX;
    private List<DenseVector> testY;
}
