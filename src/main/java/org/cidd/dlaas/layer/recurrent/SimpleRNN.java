package org.cidd.dlaas.layer.recurrent;

import org.cidd.dlaas.activation.Activation;
import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class SimpleRNN extends Recurrent {

    private INDArray w;
    private INDArray u;
    private INDArray b;
    private INDArray dw;
    private INDArray du;
    private INDArray db;
    private Activation[] activations;

    @Override
    public void recurrentConnectTo(Layer previousLayer) {
        this.w = this.initializer.handle(new int[]{this.nin, this.nout});
        this.u = this.innerInitializer.handle(new int[]{this.nout, this.nout});
        this.b = Nd4j.zeros(this.nout);
    }

    @Override
    public INDArray forward(INDArray input) {
        assert(input.length() == 3);//, 'Only support batch training.'
        this.lastInput = input;
        int nbBatch = input.shape()[0];
        int nbTimestep = input.shape()[1];
        int nbIn = input.shape()[2];

        INDArray output = Nd4j.zeros(nbBatch, nbTimestep, this.nout);

//        if (this.activations.length == 0) {
//            for (int i = 0; i < nbTimestep; i++) {
//                activationCls
//            }
//
//        }

        this.activations[0].forward(input.mmul(this.w).add(this.b));

        return null;
    }

    @Override
    public INDArray backward(INDArray input) {
        this.dw = Nd4j.zeros(this.w.shape());
        this.du = Nd4j.zeros(this.u.shape());
        this.db = Nd4j.zeros(this.b.shape());

        // hiddens.shape == (nb_timesteps, nb_batch, nb_out)
        INDArray hiddens = this.lastOutput.swapAxes(0, 1).transpose();

        return null;
    }

    @Override
    public List<INDArray> getParams() {
        return null;
    }

    @Override
    public void setParams(List<INDArray> params) {

    }

    @Override
    public List<INDArray> getGrads() {
        return null;
    }

    @Override
    public void setGrads(List<INDArray> grads) {

    }

}
