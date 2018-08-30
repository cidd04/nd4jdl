package org.cidd.dlaas.initialization;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Uniform implements Initializer {

    private double scale;

    @Override
    public INDArray handle(int[] size) {
        return Nd4j.rand(size, -scale, scale, Nd4j.getRandom());
    }
}
