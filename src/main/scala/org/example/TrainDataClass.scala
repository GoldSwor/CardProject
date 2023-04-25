package org.example

import scala.collection.mutable
class TrainPlanNode(val node:TreeNode, var cardReal:Option[Long]=None, var cardEst:Option[Long]=None,val children:Seq[TrainPlanNode]) {
  val realSelectivity=calRealSelectivity
  def setEstCard(estCard:Long)={
      this.cardEst=Some(estCard)
  }
  def getEstCard:Option[Long]={
    this.cardEst
  }
  def setRealCard(realCard:Long)={
    this.cardReal=Some(realCard)
  }
  def getRealCard:Option[Long]={
     this.cardReal
  }
  def calQError:Option[Double]={
     if(cardEst.isDefined&&cardReal.isDefined) {
      Some(math.max(cardEst.get,cardReal.get).doubleValue()/(math.min(cardEst.get,cardReal.get)+ModelUtils.EPS).doubleValue())
     }else None
  }
  def calRealSelectivity:Option[Double]={
    if(node.isInstanceOf[PlanNode]){
      val temp=node.asInstanceOf[PlanNode]
      temp.nodeType match{
        case Operator.Filter=>{
          if(children(0).cardReal.isDefined && cardReal.isDefined) {
            Some(cardReal.get.doubleValue()/(children(0).cardReal.get.doubleValue()+ModelUtils.EPS))
          }else None
        }
        case Operator.Inner|Operator.FullOuter|Operator.LeftOuter|Operator.RightOuter=>{
          if(children(0).cardReal.isDefined && children(1).cardReal.isDefined && cardReal.isDefined){
            Some(cardReal.get.doubleValue()/(children(0).cardReal.get*children(1).cardReal.get+ModelUtils.EPS).doubleValue())
          }else None
        }
        case Operator.LeftSemi|Operator.LeftAnti=>{
          if(children(0).cardReal.isDefined && cardReal.isDefined){
            Some(cardReal.get.doubleValue()/(children(0).cardReal.get.doubleValue()+ModelUtils.EPS))
          }else None
        }
        case Operator.Aggregate=>{
          if(children(0).cardReal.isDefined && cardReal.isDefined) {
            Some(cardReal.get.doubleValue()/(children(0).cardReal.get.doubleValue()+ModelUtils.EPS))
          }else None
        }
        case _=>None
      }
    }
    else None
  }
  def getEstCardBySelectivity(selectivity:Double):Option[Long]={
    if(node.isInstanceOf[PlanNode]){
      val temp=node.asInstanceOf[PlanNode]
      temp.nodeType match{
        case Operator.Filter=>{
          if(children(0).cardEst.isDefined) {
            Some((selectivity*children(0).cardEst.get).toLong)
          }else None
        }
        case Operator.Inner|Operator.FullOuter|Operator.LeftOuter|Operator.RightOuter=>{
          if(children(0).cardEst.isDefined && children(1).cardEst.isDefined){
            Some((selectivity*children(0).cardEst.get*children(1).cardEst.get).toLong)
          }else None
        }
        case Operator.LeftSemi|Operator.LeftAnti=>{
          if(children(0).cardEst.isDefined){
            Some((selectivity*children(0).cardEst.get).toLong)
          }else None
        }
        case Operator.Aggregate=>{
          if(children(0).cardEst.isDefined) {
            Some((selectivity*children(0).cardEst.get).toLong)
          }else None
        }
        case _=>None
      }
    }
    else None
  }
}
object TrainPlanTree{
  def apply(tree: Tree, labels:Seq[Long]): TrainPlanTree = new TrainPlanTree(tree,transformTreeNodeToTrainNode(tree.root.get,labels)._1)
  def transformTreeNodeToTrainNode(root:TreeNode,labels:Seq[Long],index:Int=0):(TrainPlanNode,Int)={
      if(root.isInstanceOf[PlanNode]){
        val node=root.asInstanceOf[PlanNode]
        val children=node.children.get
        val trainNodeSeq:mutable.ListBuffer[TrainPlanNode]=mutable.ListBuffer()
        var ind=index+1
        for(child <- children){
          val ans=transformTreeNodeToTrainNode(child,labels,ind)
          trainNodeSeq.append(ans._1)
          ind=ans._2
        }
        val card={
          if(labels(index)==(-1)) None
          else Some(labels(index))
        }
        (new TrainPlanNode(node,card,None,trainNodeSeq),ind)
      }
      else {
        val card={
          if(labels(index)==(-1)) None
          else Some(labels(index))
        }
        (new TrainPlanNode(root,card,None,Seq()),index+1)
      }
  }
}
class TrainPlanTree(val tree: Tree,val root:TrainPlanNode) {
  def getPlanTree:Tree=this.tree
  def getTrainRoot:TrainPlanNode=this.root
}

