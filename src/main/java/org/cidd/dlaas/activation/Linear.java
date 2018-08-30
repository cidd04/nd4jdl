package org.cidd.dlaas.activation;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class Linear extends Activation {

    public INDArray forward(INDArray input) {
        lastForward = input;
        return lastForward;
    }

    public INDArray derivative(INDArray input) {
        INDArray i = input != null ? input : lastForward;
        return Nd4j.ones(i.shape());
    }
}
