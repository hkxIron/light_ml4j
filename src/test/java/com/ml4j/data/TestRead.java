package com.ml4j.data;

import org.junit.Test;

import static com.ml4j.data.utils.FileUtils.getFileAbsolutePath;
import static com.ml4j.data.utils.FileUtils.readFile;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Email: hukexin@xiaomi.com
 * Date: 22-6-17
 * Time: 下午6:43
 */
public class TestRead {
    @Test
    public void testReadFile() throws Exception {
        String content = readFile(getFileAbsolutePath("iris.csv"));
        System.out.println("file:\n"+ content);
    }
    @Test
    public void testVector() {
        int[][] arr = {{1,2,3},{3,4,2}};
        System.out.println(arr.length);
        System.out.println(arr[0].length);
        assert arr.length==2;
        assert arr[0].length==3;
    }
}
