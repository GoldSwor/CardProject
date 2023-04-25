package org.example

import org.example.LayerKind.LayerKind

object NameSpaceUtils {

  def getNodeEmbeddingLayerMeta(node:PlanNode):(Option[String],Option[Int],Option[LayerKind])=node.nodeType  match {
    case Operator.Inner|Operator.FullOuter|Operator.LeftOuter|Operator.RightOuter|Operator.LeftSemi|Operator.LeftAnti|Operator.Cross => {
      (Some(LayerNameSpace.joinPredictEmbeddingLayerName),
        Some(LayerNum.joinPredictEmbeddingLayerNum),Some(LayerKind.MLP))
    }
    case Operator.Filter => (Some(LayerNameSpace.filterPredictEmbeddingLayerName),
      Some(LayerNum.filterPredictEmbeddingLayerNum),Some(LayerKind.MLP))
    case Operator.Aggregate => (Some(LayerNameSpace.aggregatePredictEmbeddingLayerName),
      Some(LayerNum.aggregatePredictEmbeddingLayerNum),Some(LayerKind.LSTM))
    case _ => (None,None,None)
  }

  def extractRelationNameFromColumnEmbeddingLayerNames(paramName:String):Option[String]={
    val params=paramName.split("/")
    if(params.length!=3 || !params(0).contentEquals(VarNameSpace.columnEmbeddingPrefix)) None
    else Some(params(1))
  }

  def getRelationLayerNameByRelationName(relationName:String):String={
    s"${VarNameSpace.columnEmbeddingPrefix}/${relationName}"
  }

}
