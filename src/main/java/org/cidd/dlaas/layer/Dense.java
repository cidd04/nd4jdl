package org.cidd.dlaas.layer;

import lombok.*;
import org.cidd.dlaas.activation.Activation;
import org.cidd.dlaas.initialization.Initializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Dense extends Layer {

    private int nout;
    private int nin;
    private INDArray w;
    private INDArray b;
    private INDArray dw;
    private INDArray db;
    private Initializer initializer;
    private Activation activation;
    private INDArray lastInput;

    public Dense init() {
        this.outShape = new int[2];
        this.outShape[1] = nout;
        return this;
    }

    @Override
    public void connectTo(Layer previousLayer) {
        int nin;
        if (previousLayer == null) {
            nin = this.nin;
        } else {
            int len = previousLayer.getOutShape().length;
            nin = previousLayer.getOutShape()[len - 1];
        }
        this.w = initializer.handle(new int[]{nin, this.nout});
//        this.w = Nd4j.zeros(nin, this.nout);
//        this.w = Nd4j.ones(nin, this.nout);
        this.b = Nd4j.zeros(this.nout);
    }

    @Override
    public INDArray forward(INDArray input) {
        this.lastInput = input;
        INDArray linearOut = (input.mmul(this.w)).add(this.b);
        INDArray activationOut = this.activation.forward(linearOut);
        return activationOut;
    }

    @Override
    public INDArray backward(INDArray previousGradient) {
        INDArray activationGradient = previousGradient.mul(activation.derivative(null));
        this.dw = lastInput.transpose().mmul(activationGradient);
        this.db = Nd4j.mean(activationGradient, 0);
        if (!firstLayer)
            return activationGradient.mmul(this.w.transpose());
        return null;
    }

    @Override
    public List<INDArray> getParams() {
        return Stream.of(this.w, this.b).collect(Collectors.toList());
    }

    @Override
    public void setParams(List<INDArray> params) {
        this.w = params.get(0);
        this.b = params.get(1);
    }

    @Override
    public List<INDArray> getGrads() {
        return Stream.of(this.dw, this.db).collect(Collectors.toList());
    }

    @Override
    public void setGrads(List<INDArray> grads) {
        this.dw = grads.get(0);
        this.db = grads.get(1);
    }

}
