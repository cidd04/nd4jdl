package org.cidd.dlaas.activation;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.neg;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class Sigmoid extends Activation {

    public INDArray forward(INDArray input) {
        lastForward = pow(exp(neg(input)).add(1), -1);
        return lastForward;
    }

    public INDArray derivative(INDArray input) {
        INDArray output = input != null ? forward(input) : lastForward;
        return output.mul((output.neg()).add(1));
    }

    public static void main(String[] args) {
        Activation a = new Sigmoid();
        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        INDArray r = a.forward(nd);
        System.out.println(a.forward(nd));
        System.out.println(a.derivative(null));
    }
}
