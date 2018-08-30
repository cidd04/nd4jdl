package org.cidd.dlaas.initialization;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class GlorotUniform implements Initializer {

    @Override
    public INDArray handle(long[] size) {
        Long[] decomposed = decomposeSize(size);
        double scale = Math.sqrt(6.0d / (decomposed[0].doubleValue() + decomposed[1].doubleValue()));
        return Uniform.builder().scale(scale).build().handle(size);
    }

    private Long[] decomposeSize(long[] size) {
        Long[] decomposed = new Long[2];
        if (size.length == 2) {
            decomposed[0] = size[0];
            decomposed[1] = size[1];
        } else if (size.length == 4 || size.length == 5) {
            int respectiveFieldSize = 0;
            for (int i = 2; i < size.length; i++)
                respectiveFieldSize *= size[i];
            decomposed[0] = respectiveFieldSize * size[0];
            decomposed[1] = respectiveFieldSize * size[1];
        } else {
            long respectiveFieldSize = 0;
            for (int i = 0; i < size.length; i++)
                respectiveFieldSize *= size[i];
            respectiveFieldSize = (int)Math.sqrt(respectiveFieldSize);
            decomposed[0] = respectiveFieldSize;
            decomposed[1] = respectiveFieldSize;

        }
        return decomposed;
    }
}
