package org.cidd.dlaas;

import lombok.*;
import org.cidd.dlaas.layer.Layer;
import org.cidd.dlaas.objective.Objective;
import org.cidd.dlaas.optimizer.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

@Builder
@Getter @Setter
@AllArgsConstructor
//@NoArgsConstructor
public class Model {

    private List<Layer> layers;
    private Objective objective;
    private Optimizer optimizer;

    public Model init() {
        this.layers = new ArrayList<>();
        return this;
    }

    public void add(Layer layer) {
        this.layers.add(layer);
    }

    public void compile(Objective objective, Optimizer optimizer) {
        this.objective = objective;
        this.optimizer = optimizer;
        layers.iterator().next().setFirstLayer(true);
        Layer nextLayer = null;
        for (Layer layer : layers) {
            layer.connectTo(nextLayer);
            nextLayer = layer;
        }

    }

    public void fit(INDArray x, INDArray y, int maxIter, int batchSize, boolean shuffle,
                    double validationSplit, INDArray validationData) {

        INDArray trainx = x;
        INDArray trainy = y;
        INDArray validx = null;
        INDArray validy = null;

        if (validationSplit < 1.0d && validationSplit > 0.0d) {
            int split = (int) (y.shape()[0] * validationSplit);
            validx = x.get(NDArrayIndex.interval(x.shape()[0] - split, x.shape()[0]), NDArrayIndex.all());
            validy = y.get(NDArrayIndex.interval(y.shape()[0] - split, y.shape()[0]), NDArrayIndex.all());
            trainx = x.get(NDArrayIndex.interval(0, x.shape()[0] - split), NDArrayIndex.all());
            trainy = y.get(NDArrayIndex.interval(0, y.shape()[0] - split), NDArrayIndex.all());
        } else if (validationData != null) {
            validx = validy = validationData;
        }

        int iterIndex = 0;
        while (iterIndex < maxIter) {
            iterIndex++;
            if (shuffle) {
                int seed = ThreadLocalRandom.current().nextInt(111, 1111111);
                Random random = new Random(seed);
                Nd4j.shuffle(trainx, random, 0);
                Nd4j.shuffle(trainy, random, 0);
            }

            int size = trainy.shape()[0] / batchSize;

            List<INDArray> trainLosses = new ArrayList<>();
            List<INDArray> trainPredicts = new ArrayList<>();
            List<INDArray> trainTargets = new ArrayList<>();

            for (int i = 0; i < size; i++) {
                int batchBegin = i * batchSize;
                int batchEnd = batchBegin + batchSize;
                INDArray xbatch = trainx.get(NDArrayIndex.interval(batchBegin, batchEnd), NDArrayIndex.all());
                INDArray ybatch = trainy.get(NDArrayIndex.interval(batchBegin, batchEnd), NDArrayIndex.all());

                // forward propagation
                INDArray ypred = this.predict(xbatch);

                // backward propagation
                INDArray nextGrad = this.objective.backward(ypred, ybatch);
                for (int j = this.layers.size() - 1; j >= 0; j--) {
                    nextGrad = layers.get(j).backward(nextGrad);
                }
                //
//                 get parameter and gradients
                List<INDArray> params = new ArrayList<>();
                List<INDArray> grads = new ArrayList<>();
                for (Layer layer : this.layers) {
                    params.addAll(layer.getParams());
                    grads.addAll(layer.getGrads());
                }


                // update parameters
                this.optimizer.update(layers);

                // get parameter and gradients
//                List<INDArray> params = new ArrayList<>();
//                List<INDArray> grads = new ArrayList<>();
                params.clear();
                grads.clear();
                for (Layer layer : this.layers) {
                    params.addAll(layer.getParams());
                    grads.addAll(layer.getGrads());
                }

                // got loss and predict
                trainLosses.add(this.objective.forward(ypred, ybatch));
                for (int j = 0; j < ypred.shape()[0]; j++) {
                    trainPredicts.add(ypred.getRow(j));
                    trainTargets.add(ybatch.getRow(j));
                }
            }

            System.out.printf("iter %d, train-[loss %.4f, acc %.4f]; ",
                    iterIndex,
                    this.mean(trainLosses),
                    this.accuracy(trainPredicts, trainTargets));
//                # output train status
//            runout = "iter %d, train-[loss %.4f, acc %.4f]; " % (
//            iter_idx, float(np.mean(train_losses)), float(self.accuracy(train_predicts, train_targets)))



            if (validx != null && validy != null) {

                int sizeValid = validy.shape()[0] / batchSize;

                List<INDArray> validLosses = new ArrayList<>();
                List<INDArray> validPredicts = new ArrayList<>();
                List<INDArray> validTargets = new ArrayList<>();

                for (int i = 0; i < sizeValid; i++) {
                    int batchBegin = i * batchSize;
                    int batchEnd = batchBegin + batchSize;
                    INDArray xbatch = validx.get(NDArrayIndex.interval(batchBegin, batchEnd), NDArrayIndex.all());
                    INDArray ybatch = validy.get(NDArrayIndex.interval(batchBegin, batchEnd), NDArrayIndex.all());

                    // forward propagation
                    INDArray ypred = this.predict(xbatch);

                    // got loss and predict
                    validLosses.add(this.objective.forward(ypred, ybatch));
                    validPredicts.add(ypred);
                    validTargets.add(ybatch);

                }

                // output valid status
                System.out.printf("valid-[loss %.4f, acc %.4f];\n",
                        this.mean(validLosses),
                        this.accuracy(validPredicts, validTargets));

            }

            // print runout

        }
    }

    private double mean(List<INDArray> trainLosses) {
        return trainLosses.stream().mapToDouble(v -> Nd4j.mean(v).getDouble(0)).average().getAsDouble();
    }

    public INDArray predict(INDArray x) {
        INDArray xnext = x;
        for (Layer layer : layers)
            xnext = layer.forward(xnext);
        INDArray ypred = xnext;
        return ypred;
    }

    public double accuracy(List<INDArray> outputs, List<INDArray> targets) {
        List<Double> ypredicts = new ArrayList<>();
        List<Double> ytargets = new ArrayList<>();
        for (int i = 0; i < outputs.size(); i++) {
            ypredicts.add(Nd4j.argMax(outputs.get(i), 1).getDouble(0));
            ytargets.add(Nd4j.argMax(targets.get(i), 1).getDouble(0));
        }
        double counter = 0;
        for (int i = 0; i < ytargets.size(); i++) {
            if (ytargets.get(i) == ypredicts.get(i))
                counter++;
        }
        return counter / ytargets.size();

    }

}
