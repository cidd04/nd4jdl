package org.cidd.dlaas.layer.convolution;

import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class Convolution extends Layer {

  private String nbFilter;
  private String filterSize;
  private String inputShape;
  private int stride;

  @Override
  public void connectTo(Layer layer) {

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
