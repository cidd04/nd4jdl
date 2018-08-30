package org.cidd.dlaas.optimizer;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.cidd.dlaas.layer.Layer;
import org.cidd.dlaas.util.GeneralUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

@Getter
@Setter
public class SGD extends Optimizer {

    @Builder
    public SGD(double learningRate, double clip, double decay, double minLearningRate,
               double maxLearningRate, int iterations) {
        super(learningRate, clip, decay, minLearningRate, maxLearningRate, iterations);
    }

    @Override
    public void update(List<Layer> layers) {
        // get parameter and gradients
        for (Layer layer : layers) {
            List<INDArray> params = layer.getParams();
            List<INDArray> grads = layer.getGrads();
            for (int i = 0; i < params.size(); i++) {
                INDArray a = params.get(i).sub(GeneralUtils.ndclip(grads.get(i), clip).mul(learningRate));
                params.set(i, a);
            }
            layer.setParams(params);
        }
        super.updateLearningRate();
    }
}
