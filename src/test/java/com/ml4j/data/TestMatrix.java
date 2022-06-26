package com.ml4j.data;

import com.ml4j.data.utils.GsonUtil;
import org.junit.Test;

import static com.ml4j.data.utils.GsonUtil.normalGson;

/**
 * @author: kexin
 * @date: 2022/6/23 22:13
 **/
public class TestMatrix {

    @Test
    public void testMultiply() {
        //int[][] arr = {{1, 2, 3}, {3, 4, 2}};
        DenseMatrix a = new DenseMatrix(new float[][]{
                {1, 2},
                {3, 4}});
        DenseMatrix b = new DenseMatrix(new float[][]{
                {2, 3, 4},
                {4, 2, 5}});
        DenseMatrix c = a.multiply(b);
        System.out.println(normalGson.toJson(a));
        System.out.println(normalGson.toJson(c));
        assert c.getShape()[0] == 2;
        assert c.getShape()[1] == 3;
    }

    @Test
    public void testMultiplyVector() {
        //int[][] arr = {{1, 2, 3}, {3, 4, 2}};
        DenseMatrix a = new DenseMatrix(new float[][]{{1, 2, 3}, {3, 4, 5}});
        DenseVector b = new DenseVector(new float[]{2, 3, 4});
        DenseVector c = a.multiply(b);
        System.out.println(normalGson.toJson(c));
        assert c.getShape()[0] == 2;
    }

    @Test
    public void testMultiplyToVector() {
        //int[][] arr = {{1, 2, 3}, {3, 4, 2}};
        DenseMatrix a = new DenseMatrix(new float[][]{{1, 2, 3}, {3, 4, 5}});
        DenseVector b = new DenseVector(new float[]{2, 3, 4});
        DenseVector c = a.multiply(b);
        System.out.println(normalGson.toJson(c));
    }
}
