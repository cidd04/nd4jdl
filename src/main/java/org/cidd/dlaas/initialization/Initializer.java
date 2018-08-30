package org.cidd.dlaas.initialization;

import org.nd4j.linalg.api.ndarray.INDArray;


public interface Initializer {

    INDArray handle(long[] size);

}
