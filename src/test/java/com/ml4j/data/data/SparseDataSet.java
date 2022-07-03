package com.ml4j.data.data;

import com.ml4j.data.DenseVector;
import com.ml4j.data.SparseVector;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/7/2 22:49
 **/
@Getter
@Setter
public class SparseDataSet {
    private List<String> featureNames;
    private List<SparseVector> trainX;
    private DenseVector trainY;

    private List<SparseVector> testX;
    private DenseVector testY;
}
