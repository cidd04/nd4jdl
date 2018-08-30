package org.cidd.dlaas.activation;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class Softmax extends Activation {

    public INDArray forward(INDArray input) {
        assert(input.rank() == 2);
        this.lastForward = input;
        INDArray x = input.subColumnVector(Nd4j.max(input, 1));
        INDArray ex = exp(x);
        INDArray s = ex.div(Nd4j.sum(ex, 1));
        return s;
    }

    public INDArray derivative(INDArray input) {
        INDArray i = input != null ? input : lastForward;
        return Nd4j.ones(i.shape());
    }
}
