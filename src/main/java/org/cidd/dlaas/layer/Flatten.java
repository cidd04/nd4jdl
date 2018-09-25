package org.cidd.dlaas.layer;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Flatten extends Layer {

  private int outdim;
  private long[] lastInputShape;

  @Override
  public void connectTo(Layer layer) {
    assert layer.getOutShape().length > 2;

    long toFlatten = 1;
    for (int i = this.outdim - 1; i < layer.getOutShape().length; i++) {
      toFlatten *= layer.getOutShape()[i];
    }
    this.outShape = new long[this.outdim];
    for (int i = 0; i < layer.getOutShape().length; i++) {
      this.outShape[i] = layer.getOutShape()[i];
    }
    this.outShape[this.outdim - 1] = toFlatten;
  }

  @Override
  public INDArray forward(INDArray input) {
    this.lastInputShape = input.shape();
    long[] flattenedShape = new long[this.outdim];
    for (int i = 0; i < flattenedShape.length - 1; i++) {
      flattenedShape[i] = this.lastInputShape[i];
    }
    flattenedShape[this.outdim - 1] = -1;
    return input.reshape(flattenedShape);
  }

  @Override
  public INDArray backward(INDArray input) {
    return input.reshape(this.lastInputShape);
  }

  @Override
  public List<INDArray> getParams() {
    return new ArrayList<>();
  }

  @Override
  public void setParams(List<INDArray> params) {

  }

  @Override
  public List<INDArray> getGrads() {
    return new ArrayList<>();
  }

  @Override
  public void setGrads(List<INDArray> grads) {

  }

}
