package com.ml4j.data.data;

import com.google.common.collect.Sets;
import com.ml4j.data.DenseVector;
import com.ml4j.data.SparseVector;

import java.util.*;
import java.util.stream.Collectors;

import static com.ml4j.data.utils.FileUtils.readFileByAbsolutePath;
import static com.ml4j.data.utils.FileUtils.readFile;
import static com.ml4j.data.utils.GsonUtil.normalGson;
import static com.ml4j.initializer.VectorUtils.toFloatArray;
import static com.ml4j.initializer.VectorUtils.toIntArray;

/**
 * @author: kexin
 * @date: 2022/7/2 22:54
 **/
public class DataLoader {

    public static SparseDataSet loadCensus(int totalSampleNum, float testRatio){
        String trainContents = "";
        try {
            trainContents = readFile(readFileByAbsolutePath("census/adult_train_test.data"));
        }catch (Exception ex){
            ex.printStackTrace();
        }
        String[] lines = trainContents.split("\n");
        Set<String> allCategoryValues = new HashSet<>();
        Set<String> continuesFeatureNames = Sets.newHashSet("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week");

        List<String> cols = new ArrayList<>();
        Map<String, Float> continousColumnToMax = new HashMap();
        int num =0;
        // 先加载分析数据
        for (int i = 0; i < lines.length; i++) {
            if (i == 0) { // 列名
                cols = Arrays.stream(lines[i].split(",")).map(String::trim)
                        .collect(Collectors.toList());
                System.out.println("col size:" + cols.size() + " cols:" + normalGson.toJson(cols));
                continue;
            }

            String[] arr = lines[i].split(",");
            if (arr.length != 15) {
                continue;
            }

            for (int c = 0; c < cols.size() - 1; c++) {
                if (c == 2) {
                    continue;// skip fnlwgt
                }
                String v = arr[c].trim();
                String col = cols.get(c);
                if (continuesFeatureNames.contains(col)) {
                    allCategoryValues.add(col);
                    continousColumnToMax.put(col,
                            Math.max(continousColumnToMax.getOrDefault(col, 0f),
                                    Float.parseFloat(v)));
                } else {
                    allCategoryValues.add(col + "_" + v);
                }
            }
        }

        List<String> sortedfeatValues = allCategoryValues.stream().sorted().collect(Collectors.toList());
        HashMap<String, Integer> strToIndex = new HashMap(sortedfeatValues.size());
        HashMap<Integer, String> indexToStr = new HashMap(sortedfeatValues.size());
        for (int i = 0; i < sortedfeatValues.size(); i++) {
            strToIndex.put(sortedfeatValues.get(i), i);
            indexToStr.put(i, sortedfeatValues.get(i));
        }

        System.out.println("feature values:" + normalGson.toJson(sortedfeatValues));

        List<SparseVector> trainX = new ArrayList<>();
        List<Float> trainY = new ArrayList<>();

        List<SparseVector> testX = new ArrayList<>();
        List<Float> testY = new ArrayList<>();

        String positiveLabel = ">50K";
        int positiveNum = 0;
        // 加载数据
        Random rand = new Random();
        for (int i = 0; i < lines.length; i++) {
            if (i == 0) {
                continue;
            }

            String[] arr = lines[i].split(",");
            if (arr.length != 15) {
                continue;
            }

            /*
            List<Integer> inds = new ArrayList<>();
            List<Float> vals = new ArrayList<>();
            */
            Map<Integer, Float> indexToValue = new HashMap<>();
            for (int c = 0; c < cols.size() - 1; c++) {
                if (c == 2) {
                    continue;// skip fnlwgt
                }
                String columnName = cols.get(c);
                String value = arr[c].trim();
                int index;
                float val;
                if (continuesFeatureNames.contains(columnName)) {
                    val = Float.parseFloat(value) / continousColumnToMax.get(columnName);
                    index = strToIndex.get(columnName);
                } else {
                    //vals.add(1f);
                    val = 1f;
                    index = strToIndex.get(columnName + "_" + value);
                }
                indexToValue.put(index, val);
                //inds.add(index);
            }

            float label = 0;
            if (arr[14].trim().equals(positiveLabel)) {
                label = 1;
                positiveNum++;
            } else {
                label = 0;
            }

            SparseVector x = new SparseVector(sortedfeatValues.size(), indexToValue);
            if (rand.nextDouble() < testRatio) {
                testX.add(x);
                testY.add(label);
            } else {
                trainX.add(x);
                trainY.add(label);
            }
            num++;
            if(num>=totalSampleNum){
                break;
            }
        }
        SparseDataSet dataSet = new SparseDataSet();
        dataSet.setFeatureNames(sortedfeatValues);
        dataSet.setTestX(testX);
        dataSet.setTestY(new DenseVector(toFloatArray(testY)));

        dataSet.setTrainX(trainX);
        dataSet.setTrainY(new DenseVector(toFloatArray(trainY)));

        System.out.println("positive num:"+ positiveNum + " all num:"+ (testY.size()+ trainY.size()));
        return dataSet;
    }
}
