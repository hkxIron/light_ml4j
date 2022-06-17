package com.ml4j.data;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Email: hukexin@xiaomi.com
 * Date: 22-6-17
 * Time: 下午5:54
 */
public abstract class Tensor {
    public abstract int[] getShape();
    public abstract void reshape(int[] shapes);
}
