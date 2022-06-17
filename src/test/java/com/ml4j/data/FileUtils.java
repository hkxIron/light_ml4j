package com.ml4j.data;


import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Email: hukexin@xiaomi.com
 * Date: 22-6-10
 * Time: 下午5:41
 */
public class FileUtils {
    public static String getFileAbsolutePath(String relPath) throws URISyntaxException {
        URL res = FileUtils.class.getClassLoader().getResource(relPath);
        File file = Paths.get(res.toURI()).toFile();
        return file.getAbsolutePath();
    }

    public static String readFile(String filePath) throws IOException {
        String content = com.google.common.io.Files.asCharSource(new File(filePath), StandardCharsets.UTF_8).read();
        return content;
    }
}
