package org.example

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.{DefaultScalaModule, ScalaObjectMapper}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, LSTM, Layer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.example.LayerKind.LayerKind
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.LoggerFactory

import java.io.File
import java.net.URI
import java.util
import scala.collection.mutable

object ModelUtils {
  val normalLayers=Seq(LayerNameSpace.aggregatePredictEmbeddingLayerName,
    LayerNameSpace.filterPredictEmbeddingLayerName,
    LayerNameSpace.joinPredictEmbeddingLayerName,
    LayerNameSpace.cardEstLayer,
    LayerNameSpace.nodeEmbeddingLayer)
  val seed=123
  val logger=LoggerFactory.getLogger("model load exception")
  val EPS=1e-30
  val learningRate=0.001
  val regularizationRate=1e-6
  /**
   *
   * @param jsonString:从json文件中读取的字符串，一般json文件为模型配置文件
   * @return Map:存储了从配置项名到配置项值之间的映射
   */
  def jsonStringToMap(jsonString:String):Map[String,Any]={
    val mapper = new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    mapper.readValue[Map[String, Int]](jsonString)
  }
  def initModel(relations:Seq[Relation]):mutable.Map[String,Seq[Layer]]={
    val ans=mutable.Map[String,Seq[Layer]]()
    relations.map(f=>(f.name,Seq(new DenseLayer
    .Builder()
      .nIn(f.getColumnNum)
      .nOut(f.getColumnNum<<1)
      .activation(Activation.LEAKYRELU).build(),
      new DenseLayer
      .Builder()
        .nIn(f.getColumnNum<<1)
        .nOut(ModelDim.columnEmbeddingDim)
        .activation(Activation.TANH).build()
    ))).foreach(f=>ans.put(NameSpaceUtils.getRelationLayerNameByRelationName(f._1),f._2))
    ans.put(LayerNameSpace.joinPredictEmbeddingLayerName,
      Seq(new DenseLayer.Builder().nIn(ModelDim.predictDim).nOut(ModelDim.predictEmbeddingDim).activation(Activation.LEAKYRELU).build()))
    ans.put(LayerNameSpace.filterPredictEmbeddingLayerName,
      Seq(new DenseLayer.Builder().nIn(ModelDim.predictDim).nOut(ModelDim.predictEmbeddingDim).activation(Activation.LEAKYRELU).build()))
    ans.put(LayerNameSpace.aggregatePredictEmbeddingLayerName,
      Seq(new LSTM.Builder().nIn(ModelDim.columnEmbeddingDim).nOut(ModelDim.predictEmbeddingDim).build()))
    ans.put(LayerNameSpace.nodeEmbeddingLayer,
      Seq(new LSTM.Builder().nIn(ModelDim.TLSTMInputDim).nOut(ModelDim.hiddenVectorDim).build()))
    ans.put(LayerNameSpace.cardEstLayer,
      Seq(new DenseLayer.Builder().nIn(ModelDim.hiddenVectorDim).nOut(ModelDim.hiddenVectorDim>>1).activation(Activation.LEAKYRELU).build(),
        new DenseLayer.Builder().nIn(ModelDim.hiddenVectorDim>>1).nOut(1).activation(Activation.SIGMOID).build()))
    ans
  }
  //  def initWeightForLayers(relations:Seq[Relation],normalLayers:Seq[String]):(util.Map[String,util.Map[String,INDArray]],
  //    util.Map[String,util.Map[String,INDArray]])={
  //
  //  }

  /**
   *
   * @param graph:训练完成的计算图
   * @param node:用于训练的PlanNode
   * @return:1、返回normalWeight
   * 2、返回columnEmbeddingWeight
   */
  def getWeightForLayers(graph: =>ComputationGraph,node:PlanNode):(util.Map[String,util.Map[String,INDArray]],util.Map[String,util.Map[String,INDArray]])={
    val params=graph.paramTable(true)
    val layerMeta=NameSpaceUtils.getNodeEmbeddingLayerMeta(node)
    val predictEmbeddingLayerName=layerMeta._1.get
    val normalLayerWeights=new util.HashMap[String,util.Map[String,INDArray]]()
    val columnEmbeddingLayerWeights=new util.HashMap[String,util.Map[String,INDArray]]()
    //step1:extract normalLayers' params
    val extractNormalLayerName= Seq(predictEmbeddingLayerName,LayerNameSpace.nodeEmbeddingLayer,LayerNameSpace.cardEstLayer)
    extractNormalLayerName.foreach(x=> {
      val tempMap=new util.HashMap[String,INDArray]()
      getLayerParams(x, getNormalLayerNum(x), getNormalLayerKind(x))
        .map(y => (y, params.get(y)))
        .foreach(z=>tempMap.put(z._1,z._2))
      normalLayerWeights.put(x,tempMap)
    })
    //step2:extract column embedding layers' params
    val iter=params.entrySet().iterator()
    while(iter.hasNext) {
      val entry=iter.next()
      val x=entry.getKey
      val y=entry.getValue
      val name = NameSpaceUtils.extractRelationNameFromColumnEmbeddingLayerNames(x)
      if (name.isDefined) {
        val relationName = name.get
        if (!columnEmbeddingLayerWeights.containsKey(relationName)) {
          val param = new util.HashMap[String, INDArray]()
          param.put(x, y)
          columnEmbeddingLayerWeights.put(relationName, param)
        } else {
          columnEmbeddingLayerWeights.get(relationName).put(x, y)
        }
      }
    }
    (normalLayerWeights,columnEmbeddingLayerWeights)
  }
  //layerName_W、layerName_b for dense layer
  //layerName_W、layerName_RW、layerName_b for LSTM layer
  def setWeightForLayers(normalWeights:util.Map[String,util.Map[String,INDArray]],
                         columnEmbeddingLayerWeight:util.Map[String,util.Map[String,INDArray]],
                         relationMap:mutable.Map[String,Set[String]], graph: =>ComputationGraph,node:PlanNode)={
    //first,check normal weights
    val layerMeta=NameSpaceUtils.getNodeEmbeddingLayerMeta(node)
    val predictLayerName=layerMeta._1.get
    val predictLayerNum=layerMeta._2.get
    val predictLayerKind=layerMeta._3.get

    assert(normalWeights.containsKey(LayerNameSpace.nodeEmbeddingLayer) && checkLayers(VarNameSpace.yLastOutputName,
      normalWeights.get(VarNameSpace.yLastOutputName),LayerNum.nodeEmbeddingLayerNum,LayerKind.LSTM))

    assert(normalWeights.containsKey(LayerNameSpace.cardEstLayer) &&
      checkLayers(LayerNameSpace.cardEstLayer,
        normalWeights.get(LayerNameSpace.cardEstLayer),LayerNum.cardEstLayerNum,LayerKind.MLP))

    assert(normalWeights.containsKey(predictLayerName) &&
      checkLayers(predictLayerName,
        normalWeights.get(predictLayerName),predictLayerNum,predictLayerKind))

    relationMap.keys.foreach(x=>assert(columnEmbeddingLayerWeight.containsKey(x) &&
      checkLayers(s"${VarNameSpace.columnEmbeddingPrefix}/${x}",
        columnEmbeddingLayerWeight.get(x),LayerNum.relationLayerNum,LayerKind.MLP)))

    val params=getLayerParams(LayerNameSpace.nodeEmbeddingLayer,LayerNum.nodeEmbeddingLayerNum,LayerKind.LSTM)
      .map(x=>(x,normalWeights.get(LayerNameSpace.nodeEmbeddingLayer).get(x))) ++
      getLayerParams(LayerNameSpace.cardEstLayer,LayerNum.cardEstLayerNum,LayerKind.MLP)
        .map(x => (x,normalWeights.get(LayerNameSpace.cardEstLayer).get(x))) ++
      getLayerParams(predictLayerName,predictLayerNum,predictLayerKind)
        .map(x => (x,normalWeights.get(predictLayerName).get(x))) ++
      relationMap.keys.flatMap(x => getLayerParams(s"${VarNameSpace.columnEmbeddingPrefix}/${x}",
        LayerNum.relationLayerNum,LayerKind.MLP).flatMap(y=>Seq((y,columnEmbeddingLayerWeight.get(x).get(y))))).toSeq

    val paramTable=new util.HashMap[String,INDArray]()
    params.foreach(x=>paramTable.put(x._1,x._2))
    graph.setParamTable(paramTable)
  }
  def getLayerParams(layerName:String,layerNum:Int,kind:LayerKind):Seq[String]=kind match{
    case LayerKind.MLP => getMultipleMLPLayerParams(layerName,layerNum)
    case LayerKind.LSTM => getMultipleLSTMLayerParams(layerName,layerNum)
  }
  def getNormalLayerKind(layerName:String):LayerKind={
    assert(normalLayers.exists(layerName=>true))
    if(layerName==LayerNameSpace.cardEstLayer || layerName==LayerNameSpace.joinPredictEmbeddingLayerName||
      layerName==LayerNameSpace.filterPredictEmbeddingLayerName) LayerKind.MLP
    else LayerKind.LSTM
  }
  def getNormalLayerNum(layerName:String):Int={
    assert(normalLayers.exists(layerName=>true))
    if(layerName==LayerNameSpace.cardEstLayer) LayerNum.cardEstLayerNum
    else if(layerName==LayerNameSpace.joinPredictEmbeddingLayerName) LayerNum.joinPredictEmbeddingLayerNum
    else if(layerName==LayerNameSpace.filterPredictEmbeddingLayerName) LayerNum.filterPredictEmbeddingLayerNum
    else if(layerName==LayerNameSpace.nodeEmbeddingLayer) LayerNum.nodeEmbeddingLayerNum
    else if(layerName==LayerNameSpace.aggregatePredictEmbeddingLayerName) LayerNum.aggregatePredictEmbeddingLayerNum
    else throw new Exception(s"${layerName} does not exist!")
  }
  //layerName_W、layerName_RW、layerName_b for LSTM layer
  def getMultipleLSTMLayerParams(layerName:String,layerNum:Int):Seq[String]={
    (0 until layerNum).flatMap(i => Iterable(s"${layerName}/${i}_W",s"${layerName}/${i}_RW",s"${layerName}/${i}_b"))
  }
  //layerName_W、layerName_b for dense layer
  def getMultipleMLPLayerParams(layerName:String,layerNum:Int):Seq[String]={
    (0 until layerNum).flatMap(i => Iterable(s"${layerName}/${i}_W",s"${layerName}/${i}_b"))
  }
  def checkLayers(layerName:String,layerWeights: util.Map[String,INDArray],layerNum: Int,kind:LayerKind):Boolean=kind match {
    case LayerKind.MLP => checkMultipleMLPLayers(layerName,layerWeights,layerNum)
    case LayerKind.LSTM => checkMultipleLSTMLayers(layerName,layerWeights,layerNum)
  }
  def checkMultipleLSTMLayers(layerName:String,layerWeights: util.Map[String,INDArray],layerNum:Int):Boolean= {
    if(layerWeights.size()==layerNum*3)
    {
      val keys=(0 until layerNum).flatMap(x=>Seq(s"${layerName}/${x}_W",s"${layerName}/${x}_RW",s"${layerName}/${x}_b"))
      keys.foldLeft(true)((x,y)=>x && layerWeights.containsKey(y))
    }
    else false
  }
  def checkMultipleMLPLayers(layerName:String,layerWeights:util.Map[String,INDArray],layerNum:Int):Boolean={
    if(layerWeights.size()==layerNum*2)
    {
      val keys=(0 until layerNum).flatMap(x=>Seq(s"${layerName}/${x}_W",s"${layerName}/${x}_b"))
      keys.foldLeft(true)((x,y)=>x && layerWeights.containsKey(y))
    }
    else false
  }
  /**
   *
   * @param confPath:模型配置文件在hdfs上的存储路径
   * @return Map:存储了从配置项名到配置项值之间的映射
   */
  def getConf(confPath:String) ={
    val fs=FileSystem.get(URI.create(confPath), new Configuration())
    if(fs.exists(new Path(confPath))) {
      val in = fs.open(new Path(confPath))
      val jsonString = in.readUTF()
      in.close()
      this.jsonStringToMap(jsonString)
    }
    else {
      throw new Exception(s"conf file ${confPath} not exist")
    }
  }
}
