package org.cidd.dlaas.layer.recurrent;

import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

public class BatchLSTM extends Recurrent {

    private boolean neeedGrad;
    private int forgetBiasSum;

    private INDArray allw;
    private INDArray dallw;
    private INDArray c0;
    private INDArray dc0;
    private INDArray h0;
    private INDArray dh0;
    private INDArray ifogf;
    private INDArray ifog;
    private INDArray hin;
    private INDArray ct;
    private INDArray c;

    @Override
    public void recurrentConnectTo(Layer previousLayer) {

        // init weights
        this.allw = Nd4j.zeros(this.nin + this.nout + 1, 4 * nout);

        // bias
        if (this.forgetBiasSum != 0) {
            // self.AllW[0, self.n_out: 2 * self.n_out] = self.forget_bias_num
            this.allw.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(this.nout, this.nout*2)},
                    this.forgetBiasSum);
        }

        // Weights matrices for input x
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*0, this.nout*1)},
                initializer.handle(new long[]{this.nin, this.nout}));
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*1, this.nout*2)},
                initializer.handle(new long[]{this.nin, this.nout}));
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*2, this.nout*3)},
                initializer.handle(new long[]{this.nin, this.nout}));
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*3, this.nout*4)},
                initializer.handle(new long[]{this.nin, this.nout}));

        // Weights matrices for memory cell
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*0, this.nout*1)},
                innerInitializer.handle(new long[]{this.nin, this.nout}));
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*1, this.nout*2)},
                innerInitializer.handle(new long[]{this.nin, this.nout}));
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*2, this.nout*3)},
                innerInitializer.handle(new long[]{this.nin, this.nout}));
        this.allw.put(new INDArrayIndex[]{NDArrayIndex.interval(1, this.nin+1),
                        NDArrayIndex.interval(this.nout*3, this.nout*4)},
                innerInitializer.handle(new long[]{this.nin, this.nout}));
    }

    @Override
    public INDArray forward(INDArray input) {
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
