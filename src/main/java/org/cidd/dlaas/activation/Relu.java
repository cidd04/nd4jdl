package org.cidd.dlaas.activation;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static org.nd4j.linalg.ops.transforms.Transforms.max;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class Relu extends Activation {

    public INDArray forward(INDArray input) {
        lastForward = input;
        return max(lastForward, 0);
    }

    public INDArray derivative(INDArray input) {
        INDArray result = input != null ? input : lastForward.dup();
//        INDArray res = Nd4j.zeros(i.shape());
        //res[last_forward > 0] = 1.
        BooleanIndexing.applyWhere(result, Conditions.greaterThan(0.0d), 1.0d);
        return result;
    }
}
