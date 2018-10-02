package org.cidd.dlaas.layer;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class DimShuffle extends Layer {

    private int axis;
    private long[] lastInputShape;

    @Override
    public void connectTo(Layer layer) {
        assert layer.getOutShape().length >= axis;
        this.outShape = new long[layer.getOutShape().length + 1];
        for (int i = 0, j = 0; i <  this.outShape.length; i++) {
            if (i == axis) {
                this.outShape[i] = 1;
            } else {
                this.outShape[i] = layer.getOutShape()[j];
                j++;
            }
        }
    }

    @Override
    public INDArray forward(INDArray input) {

        this.lastInputShape = input.shape();
        return Nd4j.expandDims(input, this.axis);
    }

    @Override
    public INDArray backward(INDArray input) {
        return input.reshape(this.lastInputShape);
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
