package org.cidd.dlaas.objective;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;

public class MeanSquaredError implements Objective {


    @Override
    public INDArray forward(INDArray outputs, INDArray targets) {
        INDArray a = pow(outputs.sub(targets), 2);
        INDArray b = Nd4j.sum(a, 1);
        INDArray c = Nd4j.mean(b);
        return c.mul(0.5d);
    }

    @Override
    public INDArray backward(INDArray outputs, INDArray targets) {
        return outputs.sub(targets);
    }
}
