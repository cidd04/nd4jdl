package org.cidd.dlaas.optimizer;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.cidd.dlaas.layer.Layer;

import java.util.List;

@Getter @Setter
@AllArgsConstructor
public abstract class Optimizer {

    protected double learningRate;
    protected double clip;
    protected double decay;
    protected double minLearningRate;
    protected double maxLearningRate;
    protected int iterations;

    public void updateLearningRate() {
        iterations++;
        learningRate *= (1.0d / (1.0d + (decay * iterations)));
        learningRate = Math.max(minLearningRate, Math.min(maxLearningRate, learningRate));
    }

    public abstract void update(List<Layer> layers);
}
