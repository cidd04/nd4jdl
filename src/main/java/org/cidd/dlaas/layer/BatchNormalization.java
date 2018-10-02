package org.cidd.dlaas.layer;

import org.cidd.dlaas.initialization.Initializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;

public class BatchNormalization extends Layer {

  private INDArray beta;
  private INDArray gamma;
  private INDArray epsilon;
  private Initializer betaInit;
  private Initializer gammaInit;

  @Override
  public void connectTo(Layer layer) {
    long nin = layer.getOutShape()[layer.getOutShape().length - 1];
    this.beta = this.betaInit.handle(new long[]{nin});
    this.gamma = this.gammaInit.handle(new long[]{nin});

  }

  @Override
  public INDArray forward(INDArray input) {
    return null;
  }

  @Override
  public INDArray backward(INDArray input) {
    INDArray mean = Nd4j.mean(input, 0);
    INDArray xmu = input.sub(mean);
    INDArray var = Nd4j.std(xmu, 0);
    INDArray sqrtvar = Transforms.sqrt(var.add(this.epsilon));
    INDArray ivar = Transforms.pow(sqrtvar, -1);
    INDArray xhat = xmu.mul(ivar);
    INDArray gammax = this.gamma.mul(xhat);
    INDArray out = gammax.add(this.beta);
//    this.cache =
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
