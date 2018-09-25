package org.cidd.dlaas.layer.convolution;

import org.cidd.dlaas.activation.Activation;
import org.cidd.dlaas.initialization.Initializer;
import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class Convolution extends Layer {

  private INDArray w;
  private INDArray b;
  private INDArray dw;
  private INDArray db;

  protected Initializer initializer;
  private Activation activation;

  private int nbFilter;
  private long[] filterSize;
  private long[] inputShape;
  private int stride;

  protected INDArray lastInput;
  protected INDArray lastOutput;

  @Override
  public void connectTo(Layer layer) {
    long[] inputShape;
    if (layer == null) {
      assert this.inputShape != null;
      inputShape = this.inputShape;
    } else {
      inputShape = layer.getOutShape();
    }
    assert inputShape.length == 4;

    long nbBatch = inputShape[0];
    long preNbFilter = inputShape[1];
    long preHeight = inputShape[2];
    long preWidth = inputShape[3];

    long filterHeight = this.filterSize[0];
    long filterWidth = this.filterSize[1];

    long height = ((preHeight - filterHeight) / this.stride) + 1;
    long width = ((preWidth - filterWidth) / this.stride) + 1;

    // output shape
    this.outShape = new long[4];
    this.outShape[0] = nbBatch;
    this.outShape[1] = this.nbFilter;
    this.outShape[2] = height;
    this.outShape[3] = width;

    // filters
    this.w = this.initializer.handle(new long[]{this.nbFilter, preNbFilter, filterHeight, filterWidth});
    this.b = Nd4j.zeros(this.nbFilter);
  }

  @Override
  public INDArray forward(INDArray input) {
    this.lastInput = input;

    // shape
    long nbBatch = input.shape()[0];
    long inputDepth = input.shape()[1];
    long oldImgh = input.shape()[2];
    long oldImgw = input.shape()[3];

    long filterh = this.filterSize[0];
    long filterw = this.filterSize[1];

    long newImgh = this.outShape[2];
    long newImgw = this.outShape[3];

    // init
    INDArray outputs = Nd4j.zeros(nbBatch, this.nbFilter, newImgh, newImgw);

    // convolution operation
    for (int x = 0; x < nbBatch; x++) {
      for (int y = 0; y < this.nbFilter; y++) {
        for (int h = 0; h < newImgh; h++) {
          for (int w = 0; w < newImgw; w++) {
            long hshift = h * this.stride;
            long wshift = w * this.stride;
            // patch
            //todo
          }
        }
      }
    }

    // nonlinear activation
    this.lastOutput = this.activation.forward(outputs);

    return this.lastOutput;
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
