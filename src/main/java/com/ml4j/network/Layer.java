package com.ml4j.network;

import com.ml4j.data.Tensor;
import com.ml4j.initializer.Initializer;
import com.ml4j.optimizer.Optimizer;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/**
 * @author: kexin
 * @date: 2022/6/23 22:57
 **/

public abstract class Layer implements ILayer {
  /*  @Getter
    private List<Layer> prevLayers;

    @Setter
    private List<Layer> nextLayers;*/

    public void setInput(Tensor x) {
    }

    public void setOutSize(int size) {
    }

    public void setInSize(int size) {
    }
}
