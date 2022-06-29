package com.ml4j.data;

import lombok.NoArgsConstructor;

/**
 * @author: kexin
 * @date: 2022/6/29 10:47
 **/
@NoArgsConstructor
public class SparseMatrix implements SparseTensor {
    private int[] rowIndex;
    private int[] columnIndex;
    private float[] value;
    private int rowNum;
    private int colNum;


    @Override
    public int[] getShape() {
        return new int[]{rowNum, colNum};
    }
}
