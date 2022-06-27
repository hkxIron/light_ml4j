package com.ml4j.metric;

/*
 * Created by IntelliJ IDEA.
 *
 * Author: hukexin
 * Email: hukexin@xiaomi.com
 * Date: 22-6-27
 * Time: 下午5:40
 */
public class Accuracy {
    public static float calculateAcc(int[] pred, int[] label) {
        assert pred.length == label.length;
        int num = pred.length;
        int correct = 0;
        for (int i = 0; i < num; i++) {
            if (pred[i] == label[i]) {
                correct += 1;
            }
        }
        return 1f * correct / num;
    }
}
