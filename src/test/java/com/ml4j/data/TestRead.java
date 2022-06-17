package com.ml4j.data;

import org.junit.Test;

import static com.ml4j.data.FileUtils.getFileAbsolutePath;
import static com.ml4j.data.FileUtils.readFile;

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
}
