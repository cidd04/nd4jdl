package org.cidd.dlaas.layer.pooling;

import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class MeanPooling extends Layer {

  private long[] poolSize;

  @Override
  public void connectTo(Layer layer) {
    int length = layer.getOutShape().length;
    assert 5 > length && length >= 3;
    long oldh = layer.getOutShape()[length - 2];
    long oldw = layer.getOutShape()[length - 1];
    long poolh = this.poolSize[0];
    long poolw = this.poolSize[1];
    long newh = oldh / poolh;
    long neww = oldw / poolw;

    assert oldh % poolh == 0;
    assert oldw % poolw == 0;

    this.outShape = layer.getOutShape();
    this.outShape[3] = newh;
    this.outShape[4] = neww;

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
