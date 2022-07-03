package com.ml4j.data.data;

import com.ml4j.data.SparseVector;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static com.ml4j.data.utils.GsonUtil.normalGson;

/**
 * @author: kexin
 * @date: 2022/7/2 19:11
 **/
public class TestSparse {
    @Test
    public void testSparseAdd() {
        SparseVector vec0 = new SparseVector(4, new HashMap<Integer, Float>() {{
            put(0, 0.2f);
            put(1, 3f);
        }});

        SparseVector vec1 = new SparseVector(4, new HashMap<Integer, Float>() {{
            put(1, 3.5f);
            put(3, 0.2f);
        }});

        SparseVector vec2 = (SparseVector) vec0.add(vec1, false);
        System.out.println(normalGson.toJson(vec2));

        Map<Integer, Float> map = new HashMap<Integer, Float>();
        map.put(0, 0.2f);
        map.put(1, 6.5f);
        map.put(3, 0.2f);
        assert vec2.equalsInTolerance(new SparseVector(4, map), 1e-5f);
        assert vec0.innerProduct(vec1) == 10.5f;

        System.out.println(normalGson.toJson(vec0.minus(vec1, false)));
    }
}
