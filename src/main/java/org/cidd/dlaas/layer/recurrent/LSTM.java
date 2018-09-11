package org.cidd.dlaas.layer.recurrent;

import org.cidd.dlaas.activation.Activation;
import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class LSTM extends Recurrent {

    private boolean neeedGrad;
    private int forgetBiasSum;

    private INDArray ug;
    private INDArray ui;
    private INDArray uf;
    private INDArray uo;

    private INDArray wg;
    private INDArray wi;
    private INDArray wf;
    private INDArray wo;

    private INDArray bg;
    private INDArray bi;
    private INDArray bf;
    private INDArray bo;

    private INDArray gradug;
    private INDArray gradui;
    private INDArray graduf;
    private INDArray graduo;

    private INDArray gradwg;
    private INDArray gradwi;
    private INDArray gradwf;
    private INDArray gradwo;

    private INDArray gradbg;
    private INDArray gradbi;
    private INDArray gradbf;
    private INDArray gradbo;

    private INDArray c0;
    private INDArray h0;
    private INDArray lastCell;

    public LSTM init() {
        return this;
    }

    @Override
    public void recurrentConnectTo(Layer previousLayer) {

        this.ug = initializer.handle(new long[]{this.nin, this.nout});
        this.ui = initializer.handle(new long[]{this.nin, this.nout});
        this.uf = initializer.handle(new long[]{this.nin, this.nout});
        this.uo = initializer.handle(new long[]{this.nin, this.nout});

        this.wg = initializer.handle(new long[]{this.nout, this.nout});
        this.wi = initializer.handle(new long[]{this.nout, this.nout});
        this.wf = initializer.handle(new long[]{this.nout, this.nout});
        this.wo = initializer.handle(new long[]{this.nout, this.nout});

        this.bg = Nd4j.zeros(this.nout);
        this.bi = Nd4j.zeros(this.nout);
        this.bf = Nd4j.zeros(this.nout).mul(this.forgetBiasSum);
        this.bo = Nd4j.zeros(this.nout);

        /*
        # Weights matrices for input x
        self.U_g = self.init((self.n_in, self.n_out))
        self.U_i = self.init((self.n_in, self.n_out))
        self.U_f = self.init((self.n_in, self.n_out))
        self.U_o = self.init((self.n_in, self.n_out))

        # Weights matrices for memory cell
        self.W_g = self.inner_init((self.n_out, self.n_out))
        self.W_i = self.inner_init((self.n_out, self.n_out))
        self.W_f = self.inner_init((self.n_out, self.n_out))
        self.W_o = self.inner_init((self.n_out, self.n_out))

        # Biases
        self.b_g = _zero((self.n_out,))
        self.b_i = _zero((self.n_out,))
        self.b_f = _one((self.n_out,)) * self.forget_bias_num
        self.b_o = _zero((self.n_out,))
         */

    }

    @Override
    public INDArray forward(INDArray input) {
        assert input.shape().length == 3;

        // record
        this.lastInput = input;

        // dim
        int[] nbBatch = Arrays.stream(input.shape()).mapToInt(i -> (int) i).toArray();
        int[] nbTimesteps = Arrays.stream(input.shape()).mapToInt(i -> (int) i).toArray();
        long[] nbIn = input.shape();

        // data
        INDArray output = Nd4j.zeros(nbBatch, nbTimesteps, this.nout);
        INDArray cell = Nd4j.zeros(nbBatch, nbTimesteps, this.nout);
        this.c0 = this.c0 != null ? this.c0 : Nd4j.zeros(nbBatch, nbBatch, this.nout);
        this.h0 = this.h0 != null ? this.h0 : Nd4j.zeros(nbBatch, nbBatch, this.nout);

        // forward
        for (int i = 0; i < nbTimesteps.length; i++) {
            int t = nbTimesteps[i];
            INDArray hpre = t == 0 ? this.h0 : output.getColumn(1);
            INDArray cpre = t == 0 ? this.h0 : output.getColumn(1);
            INDArray xnow = input.getColumn(1);
//            INDArray mnow =
        }

        return null;
    }

    @Override
    public INDArray backward(INDArray input) {
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
