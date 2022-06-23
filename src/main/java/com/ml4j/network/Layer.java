package com.ml4j.network;

/**
 * @author: kexin
 * @date: 2022/6/23 22:57
 **/
public interface Layer {
    public void forward();
    public void backward();
}
