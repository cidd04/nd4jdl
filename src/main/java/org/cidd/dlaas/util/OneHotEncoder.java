package org.cidd.dlaas.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class OneHotEncoder {

    public static INDArray encode(INDArray labels) {
        INDArray classes = GeneralUtils.unique(labels);
        long nbclasses = classes.shape()[0];
        INDArray oneHotLabels = Nd4j.zeros(labels.shape()[0], nbclasses);
        for (int i = 0; i < labels.shape()[0]; i++) {
            for (int j = 0; j < nbclasses; j++) {
                if (classes.getDouble(j) == labels.getDouble(i)) {
                    oneHotLabels.putScalar(i, j, 1d);
                }
            }
        }
        return oneHotLabels;
    }

    public static void decode() {


    }
//
//    public static void main(String[] args) {
//        INDArray a = Nd4j.create(new double[]{1,3,3,4},new int[]{4,1});
//        //Nd4j.create(double[],new int[]{length,1})
//        INDArray g = encode(a);
//        int x = 0;
//    }
}
