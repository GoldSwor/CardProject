package org.example
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.junit.Test
import org.junit.Assert._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters.asJavaIterableConverter
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.nd4j.autodiff.samediff.SameDiff

import java.util
class TestModel {
   @Test
   def test1= {
     val map=new util.HashMap[String,util.Map[String,String]]()
     val entry=new util.HashMap[String,String]()
     entry.put("b","c")
     map.put("a",entry)
     map.get("a").put("c","d")
     println(map.get("a").get("c"))
   }
  @Test
  def test2={
    println(PlanUtils.getTimeLineFromFileName("kk/a.b/1234.ser"))
  }
  @Test
  def testParseArrayString={
    val arrayString="[12,13,14,15]"
    println(PlanUtils.parseArrayString(arrayString))
  }
}
