package org.cidd.dlaas;

import org.cidd.dlaas.activation.Relu;
import org.cidd.dlaas.activation.Softmax;
import org.cidd.dlaas.initialization.GlorotUniform;
import org.cidd.dlaas.layer.Dense;
import org.cidd.dlaas.objective.SoftmaxCategoricalCrossEntropy;
import org.cidd.dlaas.optimizer.SGD;
import org.cidd.dlaas.util.GeneralUtils;
import org.cidd.dlaas.util.OneHotEncoder;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MlpDigits {

    @Test
    public void digitsTest() throws IOException {
        INDArray xtrain = Nd4j.readNumpy(ClassLoader
                .getSystemResource("data/digits.csv")
                .getPath(), ",");
        xtrain = xtrain.div(16d);

        INDArray ytrain = Nd4j.readNumpy(ClassLoader
                .getSystemResource("data/digitstarget.csv")
                .getPath(), ",");
        int nclasses = GeneralUtils.unique(ytrain).shape()[0];

        //model
        Model model = Model.builder().build().init();
        model.add(Dense.builder()
                .nin(64)
                .nout(500)
                .activation(Relu.builder().build())
                .initializer(GlorotUniform.builder().build())
                .build().init());
        model.add(Dense.builder()
                .nout(nclasses)
                .activation(Softmax.builder().build())
                .initializer(GlorotUniform.builder().build())
                .build().init());
        model.compile(SoftmaxCategoricalCrossEntropy.builder().build().init(),
                SGD.builder().learningRate(0.005).clip(-1).build());

        //train
        model.fit(xtrain, OneHotEncoder.encode(ytrain), 5, 64, false, 0.1, null);

    }

    @Test
    public void g() {
        Integer x = 2;
        List<Integer> l = new ArrayList<>();
        l.add(x);
        x = 3;
        int z = 1;
    }

    @Test
    public void mmul() throws IOException {
        INDArray mmul1 = Nd4j.readNumpy(ClassLoader
                .getSystemResource("data/mmul1")
                .getPath(), " ");

        INDArray mmul2 = Nd4j.readNumpy(ClassLoader
                .getSystemResource("data/mmul2")
                .getPath(), " ");


        int x = 0;
    }
}
