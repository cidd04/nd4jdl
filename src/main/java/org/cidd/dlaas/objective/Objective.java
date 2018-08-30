package org.cidd.dlaas.objective;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Objective {

    INDArray forward(INDArray outputs, INDArray targets);
    INDArray backward(INDArray outputs, INDArray targets);

}
