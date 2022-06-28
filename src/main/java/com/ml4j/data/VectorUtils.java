package com.ml4j.data;

/**
 * @author: kexin
 * @date: 2022/6/25 18:07
 **/
public class VectorUtils {
    public static boolean allEquals(int[] a, int[] b) {
        assert a.length == b.length;
        boolean equals = true;
        for (int i = 0; i < a.length; i++) {
            equals &= a[i] == b[i];
        }
        return equals;
    }

    public static float sum(float[] x) {
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i];
        }
        return sum;
    }

    public static float[] sign(float[] x, boolean inPlace) {
        float[] c;
        if (inPlace) {
            c = x;
        } else {
            c = new float[x.length];
        }

        for (int i = 0; i < x.length; i++) {
            c[i] = Math.signum(x[i]);
        }
        return c;
    }

    public static float[] abs(float[] x, boolean inPlace) {
        float[] c;
        if (inPlace) {
            c = x;
        } else {
            c = new float[x.length];
        }

        for (int i = 0; i < x.length; i++) {
            c[i] = Math.abs(x[i]);
        }
        return c;
    }

    public static int minIndex(float[] x) {
        int index = 0;
        for (int i = 0; i < x.length; i++) {
            if (x[i] < x[index]) {
                index = i;
            }
        }
        return index;
    }

    public static int maxIndex(float[] x) {
        int index = 0;
        for (int i = 0; i < x.length; i++) {
            if (x[i] > x[index]) {
                index = i;
            }
        }
        return index;
    }

    public static float[] add(float[] a, float[] b) {
        assert a.length == b.length;
        float[] c = new float[b.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + a[i];
        }
        return c;
    }

    public static float[] minus(float[] a, float[] b) {
        assert a.length == b.length;
        float[] c = new float[b.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] - a[i];
        }
        return c;
    }

    public static int[] toInt(float[] a) {
        int[] c = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            c[i] = (int) a[i];
        }
        return c;
    }

    public static float innerProduct(float[] a, float[] b) {
        assert a.length == b.length;
        float c = 0;
        for (int i = 0; i < a.length; i++) {
            c += a[i] * b[i];
        }
        return c;
    }
}
