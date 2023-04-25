package org.example

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.net.URI
import java.util
import java.util._
import scala.collection.mutable
//hdfs uri: 'hdfs://10.214.151.19:10000'
object ModelWeightPersist {
  /**
   *
   * @param uri:hdfs uri
   * @param modelPath:hdfs 模型参数文件的存储路径
   * @param user:hdfs 登录用户
   * @param normalWeights:normal Model的模型参数
   * @param relations: 列嵌入模型的模型参数
   */
  def saveWeightForLayers(uri:String,modelPath:String,user:String,normalWeights:Map[String,Map[String,INDArray]],relations:Map[String,Map[String,INDArray]])={
      val conf=new Configuration()
      val fs=FileSystem.get(URI.create(uri),conf,user)
      val normalModelRoot=s"/${modelPath}/normal"
      val relationModelRoot=s"/${modelPath}/relations"
      writeINDArrayToFile(normalModelRoot,fs,normalWeights)
      writeINDArrayToFile(relationModelRoot,fs,relations)
  }
  def writeINDArrayToFile(parentPath:String,fs:FileSystem,data:Map[String,Map[String,INDArray]])={
    val iter0=data.entrySet().iterator()
    while(iter0.hasNext){
      val entry0=iter0.next()
      val layerName=entry0.getKey
      val layerParamTable=entry0.getValue
      val  iter1=layerParamTable.entrySet().iterator()
      while(iter1.hasNext){
        val entry1=iter1.next()
        val layerParamName=entry1.getKey
        val layerParam=entry1.getValue
        val outputStream=fs.create(new Path(s"${parentPath}/${layerName}/${layerParamName}"))
        Nd4j.write(outputStream,layerParam)
      }
    }
  }
  def readINDArrayFromFile(parentPath:String,fs:FileSystem,data:Map[String,Set[String]]):Map[String,Map[String,INDArray]]={
      val paramTable=new util.HashMap[String,util.Map[String,INDArray]]()
      val iter0=data.entrySet().iterator()
      while(iter0.hasNext){
        val entry0=iter0.next()
        val layerName=entry0.getKey
        val layerParamSet=entry0.getValue
        val iter1=layerParamSet.iterator()
        val temp=new util.HashMap[String,INDArray]()
        while(iter1.hasNext){
           val layerParamName=iter1.next()
           val path=new Path(s"${parentPath}/${layerName}/${layerParamName}")
           val inputStream = fs.open(path)
           val param = Nd4j.read(inputStream)
           temp.put(layerParamName, param)
        }
        paramTable.put(layerName, temp)
      }
    paramTable
  }

  /**
   *
   * @param uri:hdfs uri
   * @param modelPath:hdfs 中模型参数的存储路径，相对user根目录
   * @param user:hdfs 登录用户
   * @param relations:要读取的列嵌入模型涉及的关系
   * @return
   * 1、返回normalWeight(ModelUtils中normalModels中的所有模型的参数）
   * 2、和relationWeight（列嵌入模型的所有参数）
   */
  def loadWeightForLayers(uri:String,modelPath:String,user:String,relations:mutable.Map[String,Relation]):
  (Map[String,Map[String,INDArray]],Map[String,Map[String,INDArray]])={
    val conf=new Configuration()
    val fs=FileSystem.get(URI.create(uri),conf,user)
    val normalModelRoot=s"/${modelPath}/normal"
    val relationModelRoot=s"/${modelPath}/relations"
    val normalWeightData=new util.HashMap[String,Set[String]]()
    val normalLayerNames=ModelUtils.normalLayers.map(x=>(x,
      ModelUtils.getLayerParams(x,ModelUtils.getNormalLayerNum(x),ModelUtils.getNormalLayerKind(x))))
    for((x,y) <- normalLayerNames){
      val temp=new util.HashSet[String]()
      for(z <- y) temp.add(z)
      normalWeightData.put(x,temp)
    }
    val normalWeight=readINDArrayFromFile(normalModelRoot,fs,normalWeightData)
    val relationModelMap=new util.HashMap[String,Set[String]]()
      relations.keys
      .map(x=>
        {
          val temp=new util.HashSet[String]()
          ModelUtils
            .getLayerParams(s"${VarNameSpace.columnEmbeddingPrefix}/${x}",LayerNum.relationLayerNum,LayerKind.MLP)
            .foreach(i=>temp.add(i))
          (x,temp)
        }).foreach(x=>relationModelMap.put(x._1,x._2))
    val relationWeight=readINDArrayFromFile(relationModelRoot,fs,relationModelMap)
    (normalWeight,relationWeight)
  }

}
