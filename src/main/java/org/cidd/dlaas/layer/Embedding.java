package org.cidd.dlaas.layer;

import org.cidd.dlaas.util.GeneralUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Embedding extends Layer {

  private long nbBatch;
  private long nbSeq;
  private boolean statik;
  private INDArray embedWords;
  private INDArray dembedWords;
  private INDArray lastInput;

  @Override
  public void connectTo(Layer layer) {
    this.outShape = new long[]{this.nbBatch, this.nbSeq, this.embedWords.shape()[1]};
  }

  @Override
  public INDArray forward(INDArray input) {
    assert input.shape().length == 2;
    this.lastInput = input;
    return this.embedWords.get(input);
  }

  @Override
  public INDArray backward(INDArray input) {
    if (!this.statik) {
      // init
      this.dembedWords = Nd4j.zeros(this.embedWords.shape());

      //
      INDArray flattenIdxs = this.lastInput.reshape(-1);
      INDArray uniqueIdxs = GeneralUtils.unique(flattenIdxs);
      INDArray flattenGrads = input.reshape(-1, this.outShape[this.outShape.length - 1]);
      for (int i = 0; i < uniqueIdxs.length(); i++) {
        //self.d_embed_words[idx] += np.sum(flatten_grads[flatten_idxs==idx], axis=0)
      }
    }
    return null;
  }

  @Override
  public List<INDArray> getParams() {
    return statik ? new ArrayList<>() : Stream.of(this.embedWords).collect(Collectors.toList());
  }

  @Override
  public void setParams(List<INDArray> params) {
    this.embedWords = params.get(0);

  }

  @Override
  public List<INDArray> getGrads() {
    return statik ? new ArrayList<>() : Stream.of(this.dembedWords).collect(Collectors.toList());
  }

  @Override
  public void setGrads(List<INDArray> grads) {
    this.dembedWords = grads.get(0);
  }

}
