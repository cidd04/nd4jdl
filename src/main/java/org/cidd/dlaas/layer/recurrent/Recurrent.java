package org.cidd.dlaas.layer.recurrent;

import org.cidd.dlaas.activation.Activation;
import org.cidd.dlaas.initialization.Initializer;
import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Recurrent extends Layer {

    protected int nout;
    protected int nin;
    protected int nbBatch;
    protected int nbSeq;
    protected Initializer initializer;
    protected Initializer innerInitializer;
    protected Activation activationCls;
    protected Activation activation;
    protected boolean returnSequence;
    protected INDArray lastInput;
    protected INDArray lastOutput;

    public Recurrent init() {
        return null;
    }

    @Override
    public void connectTo(Layer previousLayer) {
        if (previousLayer != null) {
            assert(previousLayer.getOutShape().length == 3);
            this.nin = previousLayer.getOutShape()[previousLayer.getOutShape().length - 1];
            this.nbBatch = previousLayer.getOutShape()[0];
            this.nbSeq = previousLayer.getOutShape()[1];
        } else {
            assert(this.nin != 0);
        }

        if (this.returnSequence) {
            this.outShape = new int[3];
            this.outShape[0] = this.nbBatch;
            this.outShape[1] = this.nbSeq;
            this.outShape[2] = this.nout;
        } else {
            this.outShape = new int[2];
            this.outShape[0] = this.nbBatch;
            this.outShape[1] = this.nout;
        }
        //
        this.recurrentConnectTo(previousLayer);
    }

    public abstract void recurrentConnectTo(Layer previousLayer);

}
