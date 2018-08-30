package org.cidd.dlaas.objective;

import lombok.*;
import org.cidd.dlaas.util.GeneralUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

@Getter @Setter
@Builder
@AllArgsConstructor
public class SoftmaxCategoricalCrossEntropy implements Objective {

    private double epsilon;

    public SoftmaxCategoricalCrossEntropy init() {
        this.epsilon = 1e-11;
        return this;
    }

    @Override
    public INDArray forward(INDArray outputs, INDArray targets) {
        INDArray o = GeneralUtils.ndclip(outputs, epsilon, 1 - epsilon);
        return Nd4j.mean((targets.mul(Transforms.log(o)).sum(1)).mul(-1));
    }

    @Override
    public INDArray backward(INDArray outputs, INDArray targets) {
        INDArray o = GeneralUtils.ndclip(outputs, epsilon, 1 - epsilon);
        return o.sub(targets);
    }
}
