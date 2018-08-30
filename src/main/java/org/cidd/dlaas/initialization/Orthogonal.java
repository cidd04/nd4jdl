package org.cidd.dlaas.initialization;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Orthogonal implements Initializer {

    private double gain;

    @Override
    public INDArray handle(int[] size) {
        int[] flatShape = new int[2];
        flatShape[0] = size[0];
        int prod = 1;
        for (int i = 1; i < size.length; i++)
            prod *= size[i];
        flatShape[1] = prod;
        INDArray a = Nd4j.randn(flatShape);
        int m = a.rows();
        int n = a.columns();
        INDArray u = Nd4j.create(m < n ? m : n);
        INDArray vt = Nd4j.create(n, n, 'f');
        Nd4j.getNDArrayFactory().lapack().gesvd(a, u, null, vt);
        INDArray q = Arrays.equals(u.shape(), flatShape) ? u : vt;
        q = q.reshape(size);
        q = q.mul(this.gain);
        return q;
    }
}
