package org.cidd.dlaas.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

public class GeneralUtils {

    public static INDArray unique(INDArray indArray) {
        List<Double> nodup = new ArrayList<>();
        for (int i = 0; i < indArray.shape()[0]; i++)
            nodup.add(indArray.getDouble(i));
        double[] d = nodup.stream().distinct().mapToDouble(Double::doubleValue).toArray();
        return Nd4j.create(d, new int[]{d.length,1});
    }

    public static INDArray ndclip(INDArray input, double boundary) {
        return boundary > 0 ?  Transforms.max(Transforms.min(input, boundary), -boundary) : input;
    }

    public static INDArray ndclip(INDArray input, double min, double max) {
        return Transforms.max(Transforms.min(input, max), min);
    }

}
