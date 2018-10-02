package org.cidd.dlaas.layer.pooling;

import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;

public class MaxPooling extends Layer {

    private long[] poolSize;
    private long[] inputShape;
    private INDArray lastInput;

    @Override
    public void connectTo(Layer layer) {
        int length = layer.getOutShape().length;
        assert length >= 3;
        long oldh = layer.getOutShape()[length - 2];
        long oldw = layer.getOutShape()[length - 1];
        long poolh = this.poolSize[0];
        long poolw = this.poolSize[1];
        long newh = oldh / poolh;
        long neww = oldw / poolw;

        assert oldh % poolh == 0;
        assert oldw % poolw == 0;

        this.outShape = new long[length];
        for (int i = 0; i < this.outShape.length - 2; i++) {
            this.outShape[i] = layer.getOutShape()[i];
        }
        this.outShape[length - 2] = newh;
        this.outShape[length - 1] = neww;

    }

    @Override
    public INDArray forward(INDArray input) {
        this.inputShape = input.shape();
        long poolh = this.poolSize[0];
        long poolw = this.poolSize[1];
        long newh = this.outShape[this.outShape.length - 2];
        long neww = this.outShape[this.outShape.length - 1];

        this.lastInput = input;
        INDArray outputs = Nd4j.zeros(this.inputShape.length);
        if (input.shape().length == 4) {
            long nbBatch = input.shape()[0];
            long nbAxis = input.shape()[1];
            for (int a = 0; a < nbBatch; a++) {
                for (int b = 0; b < nbAxis; b++) {
                    for (int h = 0; h < newh; h++) {
                        for (int w = 0; w < neww; w++) {
                            outputs.put(new int[]{a, b, h, w}, Nd4j.max(
                                    input.get(
                                            NDArrayIndex.point(a),
                                            NDArrayIndex.point(b),
                                            NDArrayIndex.interval(h, h + poolh),
                                            NDArrayIndex.point(w + poolw)
                                    )));
                        }
                    }
                }
            }
        }
        else if (input.shape().length == 3) {
            long nbBatch = input.shape()[0];
            for (int a = 0; a < nbBatch; a++) {
                for (int h = 0; h < newh; h++) {
                    for (int w = 0; w < neww; w++) {
                        outputs.put(new int[]{a, h, w}, Nd4j.max(
                                input.get(
                                        NDArrayIndex.point(a),
                                        NDArrayIndex.interval(h, h + poolh),
                                        NDArrayIndex.point(w + poolw)
                                )));
                    }
                }
            }
        }
        else {
            throw new RuntimeException("GG");
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
