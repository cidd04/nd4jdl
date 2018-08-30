package org.cidd.dlaas.layer;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

@Getter @Setter
public abstract class Layer {

    protected boolean firstLayer;
    protected long[] outShape;

    public abstract void connectTo(Layer layer);
    public abstract INDArray forward(INDArray input);
    public abstract INDArray backward(INDArray input);

    public abstract List<INDArray> getParams();
    public abstract void setParams(List<INDArray> params);
    public abstract List<INDArray> getGrads();
    public abstract void setGrads(List<INDArray> grads);


}
