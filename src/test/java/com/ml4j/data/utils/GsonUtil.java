package com.ml4j.data.utils;

import com.google.gson.*;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Date: 20-8-12
 * Time: 下午4:39
 */

public class GsonUtil {
    public static final Gson normalGson = new GsonBuilder().create();
    public static final Gson prettyGson = new GsonBuilder().setPrettyPrinting().create();
}