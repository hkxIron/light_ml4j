package com.ml4j.data.math;

import com.google.gson.Gson;
import com.ml4j.data.DenseMatrix;
import com.ml4j.data.DenseVector;
import com.ml4j.data.utils.GsonUtil;
import com.ml4j.math.FunctionUtils;
import org.junit.Test;

import static com.ml4j.data.VectorUtils.allEquals;
import static com.ml4j.data.VectorUtils.sum;

/**
 * @author: kexin
 * @date: 2022/6/25 15:51
 **/
public class TestFunction {
    @Test
    public void testMultiplyVector() {
        //int[][] arr = {{1, 2, 3}, {3, 4, 2}};
        //DenseMatrix a = new DenseMatrix(new float[][]{{1, 2, 3}, {3, 4, 5}});
        DenseVector b = new DenseVector(new float[]{0, 1, 2});
        float[] result = FunctionUtils.softmax(b.data());
        System.out.println(GsonUtil.normalGson.toJson(result));
        System.out.println("sum:"+GsonUtil.normalGson.toJson(sum(result)));
    }

    @Test
    public void testAllEquals(){
       int[] a = {1,2,3};
       int[] b = {1,2,3};
       assert allEquals(a,b);
    }
}
