package org.example
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.example.BufferManager.{columnEmbeddingWeight, normalWeight}
import org.nd4j.linalg.api.ndarray.INDArray
import org.example.ModelUtils._

import java.net.URI
import java.util
import scala.collection.mutable.ArrayBuffer
object BufferManager {
  //[layerName->[paramName->param]],buffer for normalWeight
  val normalWeight:util.Map[String,util.Map[String,INDArray]]=new util.HashMap[String,util.Map[String,INDArray]]()
  //[relationName->[paramName->param]],buffer for columnEmbeddingWeight
  val columnEmbeddingWeight:util.Map[String,util.Map[String,INDArray]]=new util.HashMap[String,util.Map[String,INDArray]]()
  //buffer for trainData
  val trainData:ArrayBuffer[(TrainPlanTree,String)]=ArrayBuffer()
  def apply(uri: String, user: String, normalWeightPath: String, columnEmbeddingWeightPath: String, planPath: String,
            labelPath:String): BufferManager = new BufferManager(uri, user, normalWeightPath, columnEmbeddingWeightPath, planPath,labelPath)
  def updateNormalWeight(layerName:String,layerWeights:util.Map[String,INDArray]):Unit={
    this.normalWeight.put(layerName,layerWeights)
  }
  def updateColumnEmbeddingWeight(relationName:String,columnEmbeddingWeight:util.Map[String,INDArray]):Unit={
    this.columnEmbeddingWeight.put(relationName,columnEmbeddingWeight)
  }
}
class BufferManager(val uri:String,val user:String,val normalWeightPath:String,columnEmbeddingWeightPath:String,planPath:String,labelPath:String){
  val fs=FileSystem.get(URI.create(uri),new Configuration(),user)
  def getNormalWeight(layerName:String):Option[util.Map[String,INDArray]]={
    if(!normalWeight.containsKey(layerName)){
      val normalWeightPath=this.normalWeightPath
      if(fs.exists(new Path(s"${normalWeightPath}/${layerName}"))){
        val temp=new util.HashMap[String,util.Set[String]]()
        val params=new util.HashSet[String]()
        getLayerParams(layerName,getNormalLayerNum(layerName),getNormalLayerKind(layerName)).foreach(x=>params.add(x))
        temp.put(layerName,params)
        val ans=ModelWeightPersist.readINDArrayFromFile(normalWeightPath,fs,temp)
        normalWeight.put(layerName,ans.get(layerName)) //将params参数存储到buffer中
        Some(normalWeight.get(layerName))
      }else None
    }else{
      Some(normalWeight.get(layerName))
    }
  }
  def getColumnEmbeddingWeight(relationName:String):Option[util.Map[String,INDArray]]={
    if(!columnEmbeddingWeight.containsKey(relationName)){
      val layerName=NameSpaceUtils.getRelationLayerNameByRelationName(relationName)
      if(fs.exists(new Path(s"${columnEmbeddingWeightPath}/${layerName}"))){
        val temp=new util.HashMap[String,util.Set[String]]()
        val params=new util.HashSet[String]()
        getLayerParams(layerName, LayerNum.relationLayerNum, LayerKind.MLP).foreach(x=>params.add(x))
        temp.put(relationName,params)
        val ans=ModelWeightPersist.readINDArrayFromFile(columnEmbeddingWeightPath,fs,temp)
        columnEmbeddingWeight.put(relationName,ans.get(relationName)) //将params参数存储到buffer中
        Some(columnEmbeddingWeight.get(relationName))
      }else None
    }else {
      Some(columnEmbeddingWeight.get(relationName))
    }
  }
//  def getTrainSample(sampleSize:Int,timeLine:Long=0)={
//    PlanUtils.readPlanTreeFromFile(fs,planPath,timeLine,sampleSize).map(x=>TrainPlanTree(x._1,))
//
//  }
}
