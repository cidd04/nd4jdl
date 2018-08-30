package org.cidd.dlaas.optimizer;

import lombok.*;
import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class Adam extends Optimizer {

    private double beta1;
    private double beta2;
    private double epsilon;

    private List<INDArray> ms = new ArrayList<>();
    private List<INDArray> vs = new ArrayList<>();


    @Builder
    public Adam(double learningRate, double clip, double decay, double minLearningRate,
                double maxLearningRate, int iterations,
                double beta1,
                double beta2,
                double epsilon) {
        super(learningRate, clip, decay, minLearningRate, maxLearningRate, iterations);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public void update(List<INDArray> params, List<INDArray> grads) {
        assert(params.size() == grads.size());
        this.iterations++;
        double at = learningRate * Math.sqrt(1.0d - Math.pow(beta2, iterations)) / (1.0d - Math.pow(beta1, iterations));
        if (ms.isEmpty()) {
            for (int i = 0; i < params.size(); i++)  {
                ms.add(Nd4j.zeros(params.get(i).shape()));
            }
        }
        if (vs.isEmpty()) {
            for (int i = 0; i < params.size(); i++)  {
                vs.add(Nd4j.zeros(params.get(i).shape()));
            }
        }
        for (int i = 0; i < params.size(); i ++) {
            INDArray m = (ms.get(i).mul(beta1)).add(grads.get(i).mul(1.0d - beta1));
            ms.set(i, m);

            INDArray v = (vs.get(i).mul(beta2)).add(Transforms.pow(grads.get(i), 2).mul(1.0d - beta1));
            vs.set(i, v);

            INDArray p = m.mul(at).div(Transforms.sqrt(v).add(epsilon));
            params.set(i, p);
        }

    }

    @Override
    public void update(List<Layer> layers) {

    }
}
