package org.example

import jdk.nashorn.internal.ir.WhileNode
import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.types.IntegerType
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.graph.{ElementWiseVertex, GraphVertex, LayerVertex, MergeVertex, ReshapeVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.util.MaskLayer
import org.deeplearning4j.nn.conf.layers.{ActivationLayer, DenseLayer, DropoutLayer, EmbeddingLayer, LSTM, OutputLayer}
import org.deeplearning4j.nn.conf.memory.MemoryReport
import org.deeplearning4j.nn.graph.{ComputationGraph, vertex}
import org.deeplearning4j.nn.layers.recurrent.LastTimeStepLayer
import org.deeplearning4j.nn.layers.samediff.SameDiffLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.autodiff.loss.LossReduce
import org.nd4j.autodiff.samediff.{SDVariable, SameDiff, TrainingConfig}
import org.nd4j.linalg.activations.{Activation, IActivation}
import org.nd4j.linalg.activations.impl.ActivationSigmoid
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.controlflow.compat.While
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer
import org.nd4j.linalg.api.ops.impl.reduce.same.{AMax, Sum}
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd
import org.nd4j.linalg.api.ops.impl.transforms.strict.Cos
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.loss
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.learning.config.{Adam, Sgd}
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.util
object Train  {
  def main(args: Array[String]): Unit = {
//   val a=Nd4j.linspace(1,25,25).reshape(Array[Int](1,5,5))
//    val b=a.putScalar(Array[Int](0,1,1),250)
//    val c=a.get(NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.interval(0,2,4))
//    a.put(Array[INDArrayIndex](NDArrayIndex.point(0),NDArrayIndex.point(1),NDArrayIndex.interval(0,5)),Nd4j.ones(5))
//    println(a.getDouble(0,0,1))
//   val grah=new NeuralNetConfiguration.Builder()
//     .updater(new Sgd(0.01))
//     .graphBuilder()
//     .addInputs("a","b")
//     .addVertex("add",new ElementWiseVertex(ElementWiseVertex.Op.Add),"a","b")
//     .addLayer("dense",new DenseLayer.Builder().nIn(10).nOut(20).activation(Activation.RELU).hasBias(false).hasLayerNorm(true).build(),"add")
//     .addLayer("aout",new ActivationLayer.Builder().activation(Activation.RELU).build(),"dense")
//     .addLayer("dropout",new DropoutLayer.Builder().dropOut(0.9).build(),"aout")
//     .build()
//    val layer:MultiLayerConfiguration=new NeuralNetConfiguration.Builder()
//      .updater(new Sgd(0.01))
//      .list()
//      .layer(new LSTM.Builder().nIn(10).nOut(20).build())
//      .build()
//    val sameDiff=SameDiff.create()
//    val input=sameDiff.placeHolder("input",DataType.FLOAT,-1,3)
//    val output=sameDiff.placeHolder("output",DataType.FLOAT,2,1)
//    val numSamples=sameDiff.placeHolder("num",DataType.INT32,0)
//    val weight=sameDiff.`var`("w1",2,3)
//    val bias=sameDiff.`var`("b1",2,1)
//    val k=sameDiff.nn.linear("output",input,weight,bias)
//    val loss=sameDiff.loss.meanSquaredError(output,k,numSamples,LossReduce.MEAN_BY_WEIGHT)
//    sameDiff.setOutputs("output")
//    sameDiff.setLossVariables(loss)
//    val conf=new TrainingConfig.Builder().l2(0.00001).updater(new Adam(0.001)).build()
//    sameDiff.setTrainingConfig(conf)
//    sameDiff.setOutputs("output")
//   val sd=SameDiff.create()
//   sd.withNameScope("aa")
//   val input=sd.placeHolder("input",DataType.FLOAT,6)
//   val w1 = sd.`var`("w1",Nd4j.rand(6, 10))
//   val b1 = sd.`var`( Nd4j.rand(10))
//   val w2 = sd.`var`( Nd4j.rand(10, 5))
//   val b2 = sd.`var`(Nd4j.rand(5))
//    val linear1 = sd.nn.linear(input, w1, b1)
//    val dense1 = sd.nn.tanh(linear1)
//    val linear2 = sd.nn.linear(dense1, w2, b2)
//    val dense2 = sd.nn.relu(linear2, 1e-30)
//    val ff:Float=1
//    sd.withNameScope("bb")
//    val dense3 = sd.constant(ff).sub("dense3",dense2)
//
//   val a=new util.HashMap[String,INDArray]()
//    a.put("aa/input",Nd4j.rand(10,6))
////    val b=new util.ArrayList[String]()
////    b.add("dense3")
////    val c=new util.ArrayList[String]()
////    c.add("w1")
//
//   sd.calculateGradients(a,"aa/w1")
//    println(sd.grad("aa/w1").getArr)
//   // println(sd.getArrForVarName("dense3"))
//    a.clear()
//    a.put("aa/input",Nd4j.zeros(5,6))
//    sd.calculateGradients(a,"aa/w1")
//    println(sd.grad("aa/w1").getArr)
//    sd.rnn.
//    println(sd.getArrForVarName("dense3"))
//    println(sd.grad("w1").getArr)
    //sd.calculateGradientsAndOutputs(a,util.ArrayList[String])
 val graph=new NeuralNetConfiguration.Builder()
  .weightInit(WeightInit.XAVIER)
  .updater(new Adam(0.01))
  .graphBuilder()
  .addInputs("in")
  .addLayer("b",new DenseLayer.Builder().nIn(10).nOut(5).activation(Activation.SIGMOID).build(),"in")
  .addLayer("c",new OutputLayer.Builder().nIn(5).nOut(1).activation(Activation.RELU).lossFunction(LossFunctions.LossFunction.MSE).build(),"b")
//  .addLayer("c",new LastTimeStep(new LSTM.Builder().nIn(3).nOut(2).build()),"in")
//  .addLayer("d",new DenseLayer.Builder().nIn(2).nOut(1).build(),"c")
//  .addLayer("d",new EmbeddingLayer.Builder().nIn(10).nOut(20).activation(Activation.TANH).build(),"in")
  .setOutputs("c")
  .build()
    val net=new ComputationGraph(graph)
    net.init()
//    val params=new util.HashMap[String,INDArray]()
//    params.put("b_W",Nd4j.rand(10,5))
//    params.put("b_b",Nd4j.rand(1,5))
//    params.put("c_W",Nd4j.rand(5,1))
//    params.put("c_b",Nd4j.rand(1,1))
//    println(params)
//    net.setParamTable(params)
//   println(net.paramTable())
    net.setInputs()
   val map=net.feedForward(Array[INDArray](Nd4j.zeros(3,10)),true)
   // net.update(net.calcBackpropGradients(false,false,Nd4j.zeros(3,1)))
    println(map)
    println(net.paramTable())
//   println(dense3.getArr)
//   sd.clearPlaceholders(true)
//   a.put("input",Nd4j.rand(5,6))
//   sd.output(a,"dense3")
//   println(sd.getArrForVarName("dense3"))
//    println(sd.variables())
//    val ops=sd.ops()
//    for(op <- ops) println(op.opName(),op.args().mkString(","),op.outputs())
//    val config=new TrainingConfig.Builder().l2(1e-5).updater(new Adam(0.001))
//      .dataSetFeatureMapping("input")
//      .dataSetLabelMapping("label")
//      .build()
//    sd.setTrainingConfig(config)
//     println(sd.scalar("a",1).getShape.mkString(","))
//    val s=Array[Float](1,2,3)
//    println(s)
//    val a=Nd4j.create(Array[Float](1,2,3),Array[Int](3))
//    val b=Nd4j.zeros(NodeEncode.columnEmbeddingDim)
//    a.put(Array[INDArrayIndex](NDArrayIndex.interval(0,3)),b)
//    println(a)
  }
}





























