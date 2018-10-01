package org.cidd.dlaas.layer.convolution;

import org.cidd.dlaas.activation.Activation;
import org.cidd.dlaas.initialization.Initializer;
import org.cidd.dlaas.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
            INDArray patch = input.get(
              NDArrayIndex.point(x),
              NDArrayIndex.all(),
              NDArrayIndex.interval(hshift, hshift + filterh),
              NDArrayIndex.interval(wshift, wshift + filterw));
            INDArray sum = Nd4j.sum(patch.mul(this.w.getColumn(y).add(this.b.getColumn(y))));
            outputs.put(new int[]{x, y, h, w}, sum);
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
    // shape
    assert input.shape().length == this.lastOutput.shape().length;
    long nbBatch = this.lastInput.shape()[0];
    long inputDepth = this.lastInput.shape()[1];
    long oldImgh = this.lastInput.shape()[2];
    long oldImgw = this.lastInput.shape()[3];

    long filterh = this.filterSize[0];
    long filterw = this.filterSize[1];

    long newImgh = this.outShape[2];
    long newImgw = this.outShape[3];

    // gradients
    this.dw = Nd4j.zeros(this.w.shape());
    this.db = Nd4j.zeros(this.b.shape());
    INDArray delta = input.mul(this.activation.derivative(null));

    // dw
    for (int r = 0; r < this.nbFilter; r++) {
      for (int t = 0; t < inputDepth; t++) {
        for (int h = 0; h < filterh; h++) {
          for (int w = 0; w < filterw; w++) {
            INDArray inputWindow = this.lastInput.get(
              NDArrayIndex.all(),
              NDArrayIndex.point(t),
              NDArrayIndex.interval(h, this.stride, oldImgh - filterh + h + 1),
              NDArrayIndex.interval(w, this.stride, oldImgw - filterw + w + 1)
            );
            INDArray deltaWindow = delta.get(NDArrayIndex.all(), NDArrayIndex.point(r));
            this.dw.put(new int[]{r, t, h, w}, Nd4j.sum(inputWindow.mul(deltaWindow)).div(nbBatch));
          }
        }
      }
    }

    // db
    for (int r = 0; r < this.nbFilter; r++) {
      this.db.put(r, Nd4j.sum(delta.get(NDArrayIndex.all(), NDArrayIndex.point(r))).div(nbBatch));
    }

    // dx
    if (!this.firstLayer) {
      INDArray layerGrads = Nd4j.zeros(this.lastInput.shape());
      for (int b = 0; b < nbBatch; b++) {
        for (int r = 0; b < this.nbFilter; b++) {
          for (int t = 0; b < inputDepth; b++) {
            for (int h = 0; b < newImgh; b++) {
              for (int w = 0; b < newImgw; b++) {
                long hshift = h * this.stride;
                long wshift = w * this.stride;
                INDArrayIndex[] indices = new INDArrayIndex[]{
                  NDArrayIndex.point(b),
                  NDArrayIndex.point(t),
                  NDArrayIndex.interval(hshift, hshift + filterh),
                  NDArrayIndex.interval(wshift, wshift + filterw)
                };
                INDArray currentLayerGrad = layerGrads.get(indices);
                currentLayerGrad = currentLayerGrad.add(this.w.get(NDArrayIndex.point(r), NDArrayIndex.point(t))
                  .mul(delta.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(r),
                    NDArrayIndex.point(h),
                    NDArrayIndex.point(w)
                )));
                layerGrads.put(indices, currentLayerGrad);
              }
            }
          }
        }
      }
      return layerGrads;
    }
    return null;
  }

  @Override
  public List<INDArray> getParams() {
    return Stream.of(this.w, this.b).collect(Collectors.toList());
  }

  @Override
  public void setParams(List<INDArray> params) {
    this.w = params.get(0);
    this.b = params.get(1);
  }

  @Override
  public List<INDArray> getGrads() {
    return Stream.of(this.dw, this.db).collect(Collectors.toList());
  }

  @Override
  public void setGrads(List<INDArray> grads) {
    this.dw = grads.get(0);
    this.db = grads.get(1);
  }

}
