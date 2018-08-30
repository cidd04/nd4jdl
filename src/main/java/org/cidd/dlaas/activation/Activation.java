package org.cidd.dlaas.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Activation {

    protected INDArray lastForward;

    public abstract INDArray forward(INDArray input);
    public abstract INDArray derivative(INDArray input);


}
