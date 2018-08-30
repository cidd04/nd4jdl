package org.cidd.dlaas.activation;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class Tanh extends Activation {

    public INDArray forward(INDArray input) {
        lastForward = tanh(input);
        return lastForward;
    }

    public INDArray derivative(INDArray input) {
        INDArray output = input != null ? forward(input) : lastForward;
        return (pow(output, 2).neg()).add(1);
    }
}
