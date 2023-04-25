package org.example
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder
import org.deeplearning4j.nn.conf.graph.{ElementWiseVertex, MergeVertex, ReshapeVertex, ScaleVertex, StackVertex, UnstackVertex}
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.{DenseLayer, LSTM, Layer, OutputLayer}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scala.collection.mutable
class Model {
  var modelMap:mutable.Map[String,Seq[Layer]]=mutable.Map[String,Seq[Layer]]()
  def initModels(models:mutable.Map[String,Seq[Layer]])={
    this.modelMap=models
  }
  def getModel(modelName:String):Seq[Layer]={
    this.modelMap(modelName)
  }
  def addModel(modelName:String,layers:Seq[Layer])={
    this.modelMap(modelName)=layers
  }

  def getRelationNameForColumn(columnName:String):String={
    val qualifierName=columnName.split(".")
    //val unQualifierColumnName=qualifierName.last
    //remove last column name,reserve prefix indicate the relation column belongs
    qualifierName.zipWithIndex.filter(x=>(x._2!=qualifierName.length-1)).map(x=>x._1).mkString(".")
}
  def mergeRelationMap(oneMap:mutable.Map[String,mutable.Set[String]],otherMap:mutable.Map[String,mutable.Set[String]]):mutable.Map[String,mutable.Set[String]]={
    for((k,v)<-otherMap){
      if(oneMap.contains(k)){
        oneMap(k)=oneMap(k).union(v)
      }
      else oneMap(k)=v
    }
    oneMap
  }
  def mergeAtomicExpressionMap(oneMap:mutable.Map[String,Seq[String]],otherMap:mutable.Map[String,Seq[String]]):mutable.Map[String,Seq[String]]={
     for((k,v)<-otherMap){
       assert(!oneMap.contains(k)) //检测两个atomicExpressionMap是否包含相同的键
       oneMap.put(k,v)
     }
    oneMap
  }
  def mergeAttributeIntoRelation(columnName:String):mutable.Map[String,mutable.Set[String]]={
      val relation=getRelationNameForColumn(columnName)
      mutable.Map[String,mutable.Set[String]](relation->mutable.Set(columnName))
  }
  def mergeCondMap(oneMap:mutable.Map[Cond,String],otherMap:mutable.Map[Cond,String]):mutable.Map[Cond,String]={
       for((k,v)<-otherMap){
         assert(!oneMap.contains(k))
         oneMap.put(k,v)
       }
    oneMap
  }
  /**
   *
   * @param cond：传入某个PlanNode的条件cond
   * @param relations：传入查询计划所涉及的关系，映射为：关系名->Relation类
   * @param exprId:传入最小的可用的id,用于给计算图的中间结果命名，默认为0
   * @return 1、返回计算图要传入的参数，映射为：传入参数名->参数值
   * 2、返回嵌入的列的映射，映射为：关系名->Set(列名）
   * 3、返回atomic expression,映射为：atomic expression name->[column_name , exp_operator , column_name | exp_operand]
   * 4、返回CondNode到atomic_expression_name的映射
   * 5、返回可用的最小的exprId(unusedId)
   */
  def collectInputCondEncode(cond:Cond,relations:mutable.Map[String,Relation],exprId:Int=0):
  (mutable.Map[String,INDArray],mutable.Map[String,mutable.Set[String]], mutable.Map[String,Seq[String]],mutable.Map[Cond,String],Int)=cond.getCondType match{
  case LogicCondType=>{
    if(cond.children.isDefined) {
      var ans=mutable.Map[String,INDArray]()
      var id=exprId
      var relationMap=mutable.Map[String,mutable.Set[String]]()
      var atomicExpressionMap=mutable.Map[String,Seq[String]]() //每个atomic_expression_name都不同
      var condMap=mutable.Map[Cond,String]()
      for(i <- 0 until cond.children.get.length){
        val child=cond.children.get(i)
        val (inputMap,tempRelationMap,tempAtomicExpressionMap,tempCondMap,unusedId)=collectInputCondEncode(child,relations,id)
        ans=ans ++ inputMap
        id=unusedId
        relationMap=mergeRelationMap(relationMap,tempRelationMap)
        atomicExpressionMap=mergeAtomicExpressionMap(atomicExpressionMap,tempAtomicExpressionMap)
        condMap=mergeCondMap(condMap,tempCondMap)
      }
      (ans,relationMap,atomicExpressionMap,condMap,id)
    }
    else throw new Exception("logical cond has no children error")
  }
  case ArithmeticCondType=>{
    val condName=VarNameSpace.condPrefix+Integer.toString(cond.getCondIdx)
    val ans=mutable.Map[String,INDArray](condName->NodeEncode.encodeArithmeticCond(cond))
    if(cond.children.isDefined) {
      val children = cond.children.get
      if (children.length == 2) {
        val leftChild = children(0)
        val rightChild = children(1)
        if (leftChild.condType.equals(Condition.Attribute) && rightChild.condType.equals(Condition.Attribute)) {
          val leftRelation = leftChild.getRelationForCond(relations)
          val rightRelation = rightChild.getRelationForCond(relations)
          ans.put(leftChild.name.get, NodeEncode.encodeColumn(leftChild, leftRelation))
          ans.put(rightChild.name.get, NodeEncode.encodeColumn(rightChild, rightRelation))
          val atomicExpressionMap= mutable.Map[String,Seq[String]](s"${VarNameSpace.expPrefix}${exprId}"->Seq(s"${VarNameSpace.columnEmbeddingPrefix}/${leftChild.name.get}",condName,s"${VarNameSpace.columnEmbeddingPrefix}/${rightChild.name.get}"))
          //返回值
          (ans, mergeRelationMap(mergeAttributeIntoRelation(leftChild.name.get),
            mergeAttributeIntoRelation(rightChild.name.get)), atomicExpressionMap, mutable.Map[Cond,String](cond->s"${VarNameSpace.expPrefix}${exprId}"),
            exprId+1)
        }
        else if (leftChild.condType.equals(Condition.Attribute) && rightChild.condType.equals(Condition.Literal)) {
          val relation = leftChild.getRelationForCond(relations)
          ans.put(leftChild.name.get, NodeEncode.encodeColumn(leftChild, relation))
          val name = VarNameSpace.expPrefix + Integer.toString(exprId)
          ans.put(name, NodeEncode.encodeLiteral(rightChild, relation))
          val atomicExpressionMap= mutable.Map[String,Seq[String]](s"${VarNameSpace.expPrefix}${exprId+1}"->
            Seq(s"${VarNameSpace.columnEmbeddingPrefix}/${leftChild.name.get}",condName,name))
          (ans, mergeAttributeIntoRelation(leftChild.name.get), atomicExpressionMap,
            mutable.Map[Cond,String](cond->s"${VarNameSpace.expPrefix}${exprId+1}"),exprId + 2)
        }
        else throw new Exception("there is arithmetic cond has two children type literal")
      }
      else if (children.length == 1) {
        val child = children(0)
        if (child.condType.equals(Condition.Attribute)) {
          val relation = child.getRelationForCond(relations)
          ans.put(child.name.get, NodeEncode.encodeColumn(child, relation))
          val atomicExpressionMap= mutable.Map[String,Seq[String]](s"${VarNameSpace.expPrefix}${exprId}"->
            Seq(s"${VarNameSpace.columnEmbeddingPrefix}/${child.name.get}", condName,VarNameSpace.zeroEncoding))
          ans.put(VarNameSpace.zeroEncoding,Nd4j.zeros(1,ModelDim.columnEmbeddingDim))
          (ans, mergeAttributeIntoRelation(child.name.get), atomicExpressionMap,
           mutable.Map[Cond,String](cond->s"${VarNameSpace.expPrefix}${exprId}") ,exprId+1)
        }
        else throw new Exception("there is arithmetic cond has one child type literal")
      } else throw new Exception(s"there is arithmetic cond has ${cond.children.get.length} children")
    }
    else throw new Exception(s"there is arithmetic cond ${cond.condType} has no children")
  }
  case _=> throw  new Exception("parse error")
}

  /**
   *
   * @param condSeq:传入Seq[Cond],适用于Aggregate、Project算子
   * @param relations:传入查询计划所涉及的关系，映射为：关系名->Relation类
   * @param exprId:传入最小的可用的id,用于给计算图的中间结果命名，默认为0
   * @return 1、返回计算图要传入的参数，映射为：传入参数名->参数值
   *         2、返回嵌入的列的映射，映射为：关系名->Set(列名）
   *         3、返回可用的最小的exprId(unusedId)
   */
  def collectInputCondSeqEncode(condSeq:Seq[Cond],relations:mutable.Map[String,Relation],exprId:Int=0):(mutable.Map[String,INDArray],mutable.Map[String,mutable.Set[String]],Int)={
      assert(condSeq.foldLeft(true)((x,y)=>x&&y.getCondType.equals(Condition.Attribute))) //判断condSeq中的每个cond是否都是attribute类型
      val inputMap=condSeq.foldLeft(mutable.Map[String,INDArray]())((x,y)=>x++mutable.Map(y.name.get->NodeEncode.encodeColumn(y,y.getRelationForCond(relations))))
      val relationMap=condSeq.foldLeft(mutable.Map[String,mutable.Set[String]]())((x,y)=>mergeRelationMap(x,mergeAttributeIntoRelation(y.name.get)))
      (inputMap,relationMap, exprId)
  }

  /**
   *
   * @param node:传入待构建计算图的PlanNode节点（查询计划树上的每个PlanNode的计算图都不同）
   * @param relations：传入查询计划树所涉及的关系
   * @return
   * 1、返回该PlanNode要传入计算图的变量
   * 2、返回嵌入的列的映射，映射为：关系名->Set(列名）
   * 3、返回atomic expression name 到 [operand,operator,operand]的映射
   * 4、返回cond到atomic expression name的映射
   * 5、返回可用的最小的exprId(unusedId)
   */
  def collectInputEncode(node:PlanNode,relations:mutable.Map[String,Relation],yLast:INDArray):
  (mutable.Map[String,INDArray] ,mutable.Map[String,mutable.Set[String]], mutable.Map[String,Seq[String]], mutable.Map[Cond,String],Int)={
      val ans=mutable.Map[String,INDArray]()
      ans.put(VarNameSpace.operatorName,NodeEncode.encodeOperator(node))
      ans.put(VarNameSpace.yLastInputName,yLast)
      if(node.cond.isDefined)
      {
         if(node.cond.get.length==1)
         {
           val (inputMap ,relationMap , atomicExpressionMap, condMap, exprId)=collectInputCondEncode(node.cond.get(0),relations)
           (ans++inputMap , relationMap , atomicExpressionMap, condMap, exprId)
         }
         else if(node.cond.get.length==0) throw new Exception("if node with no cond,the cond must set to be None")
         else
         {
           val (inputMap,relationMap,exprId)=collectInputCondSeqEncode(node.cond.get, relations)
           (ans++inputMap, relationMap, mutable.Map[String,Seq[String]](), mutable.Map[Cond,String](), exprId)
         }
      }
      else
      {
          ans.put(VarNameSpace.noCondName,Nd4j.zeros(1,ModelDim.predictEmbeddingDim)) //若算子不包含任何条件表达式，则cond_encode为零向量，可能用不上
         (ans,mutable.Map[String,mutable.Set[String]](),mutable.Map[String,Seq[String]](), mutable.Map[Cond,String](), 0)
      }
}

  /**
   *
   * @param firstVarName:输入网络第一层的变量名
   * @param layerName:待插入网络名，
   * 如：card_est_layers,
   * join_predict_embedding_layers,
   * filter_predict_embedding_layers,
   * aggregate_predict_embedding_layers,
   * column_embedding_layers,
   * node_embedding_layers
   * @param layers:输入要插入的网络层
   * @param conf：GrpahBuilder，构建计算图的类
   * @return 返回最后一层网络输出的变量名
   */
  def insertMultipleLayerIntoComputationGraph(firstVarName:String,layerName:String,layers:Seq[Layer],conf: =>GraphBuilder):String={
    var inputVar=firstVarName
    for((layer,layerNum) <- layers.zipWithIndex){
      val outputVar=s"${layerName}/${Integer.toString(layerNum)}"
      conf.addLayer(outputVar,layer,inputVar)
      inputVar=outputVar
    }
    inputVar
  }
  /**
   * 以下函数可以建立Seq2Seq的计算图
   * params:
   * 1、inputSequence:输入每层LSTM的输入变量名
   * 2、inputDimSeq：输入每层输入向量的维度
   * 3、outputDimSeq:输出每层输出向量的维度
   * 4、layerName:该Seq2Seq model的名字
   * 5、每层lstm层的实例
   * 6、conf:GraphBuilder,要加入Seq2Seq层的计算图
   * return:
   * 1、返回最后层的输出变量名
   */
  def insertMultipleLSTMLayerIntoComputationGraph(inputSequence:Seq[Seq[String]],inputDimSeq:Seq[Int],outputDimSeq:Seq[Int],layerName:String,layers:Seq[Layer],conf: =>GraphBuilder):String={
    assert(inputSequence.length==layers.length && inputDimSeq.length==layers.length && outputDimSeq.length==layers.length)

    for((layer,layerNum)<- layers.zipWithIndex){
      if(layerNum==0) {
        val input = inputSequence(layerNum)
        conf.addVertex(s"${layerName}/preprocess_${layerNum}",
          new ReshapeVertex('f', Array[Int](1, inputDimSeq(layerNum), input.length), null), input: _*)
        conf.addLayer(s"${layerName}/${layerNum}", new LastTimeStep(layer), s"${layerName}/preprocess_${layerNum}")
      }else{
        val input=inputSequence(layerNum).flatMap(x=>Seq(x,s"${layerName}/${layerNum-1}"))
        conf.addVertex(s"${layerName}/preprocess_${layerNum}",
          new ReshapeVertex('f',Array[Int](1,inputDimSeq(layerNum)+outputDimSeq(layerNum-1),input.length>>1),null),input:_*)
        conf.addLayer(s"${layerName}/${layerNum}",new LastTimeStep(layer),s"${layerName}/preprocess_${layerNum}")
      }
    }

    s"${layerName}/${layers.length-1}"
  }
  def buildComputationGraphForPlanNode(node:PlanNode, relations:mutable.Map[String,Relation], yLast:INDArray):(ComputationGraph,String,String)={
    val conf=new NeuralNetConfiguration.Builder()
      .seed(ModelUtils.seed)
      .l2(ModelUtils.regularizationRate)
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam(ModelUtils.learningRate))
      .graphBuilder()
    val ans=collectInputEncode(node, relations,yLast)
    val inputMap=ans._1
    val relationMap=ans._2
    val atomicExpressionMap=ans._3
    val condMap=ans._4
    val exprId=ans._5
    conf.addInputs(inputMap.keys.toSeq:_*)

    //step1:embedding all columns exist in the plan tree
    for((k,v)<-relationMap)
    {
      val columns=v.toSeq
      val step=1
      conf.addVertex(s"${VarNameSpace.columnEmbeddingPreprocessPrefix}/${k}",new StackVertex(),columns:_*)
      val layerName=s"${VarNameSpace.columnEmbeddingPrefix}/${k}"
      val layerOutput=insertMultipleLayerIntoComputationGraph(s"${VarNameSpace.columnEmbeddingPreprocessPrefix}/${k}",layerName,getModel(layerName),conf)
      for((col,from) <- columns.zipWithIndex)
      {
          conf.addVertex(s"${VarNameSpace.columnEmbeddingPrefix}/${col}",new UnstackVertex(from,step),layerOutput)
      }
    }

    //step2:embedding all atomic expression in the condition tree
    for((k,v)<-atomicExpressionMap){
        conf.addVertex(k,new MergeVertex(),v:_*)
    }

   val outputVar= node.nodeType match {
      case Operator.Inner|Operator.FullOuter|Operator.LeftOuter|Operator.RightOuter|Operator.LeftSemi|Operator.LeftAnti|Operator.Cross =>{
        val layerName = LayerNameSpace.joinPredictEmbeddingLayerName
        if(node.cond.isDefined)
        {
          val atomicExpressionSeq=atomicExpressionMap.keys.toSeq
          conf.addVertex(VarNameSpace.atomicExpressionPrefix,new StackVertex(), atomicExpressionSeq:_*)
          val outputVar=insertMultipleLayerIntoComputationGraph(VarNameSpace.atomicExpressionPrefix,layerName,getModel(layerName),conf)
          for((exp,from)<-atomicExpressionSeq.zipWithIndex) {
            conf.addVertex(s"${VarNameSpace.atomicExpressionPrefix}/${exp}",new UnstackVertex(from,1),outputVar)
          }
          val (_,exprOutput)=buildComputationGraphForCond(node.cond.get(0),relations,condMap,conf,exprId)
          exprOutput
        }
        else VarNameSpace.noCondName
      }
      case Operator.Filter=>{
        val layerName=LayerNameSpace.filterPredictEmbeddingLayerName
        if (node.cond.isDefined)
        {
          val atomicExpressionSeq=atomicExpressionMap.keys.toSeq
          conf.addVertex(VarNameSpace.atomicExpressionPrefix,new StackVertex(), atomicExpressionSeq:_*)
          val outputVar=insertMultipleLayerIntoComputationGraph(VarNameSpace.atomicExpressionPrefix,layerName,getModel(layerName),conf)
          for((exp,from)<-atomicExpressionSeq.zipWithIndex) {
            conf.addVertex(s"${VarNameSpace.atomicExpressionPrefix}/${exp}",new UnstackVertex(from,1),outputVar)
          }
          val (_,exprOutput)=buildComputationGraphForCond(node.cond.get(0),relations,condMap,conf,exprId)
          exprOutput
        }
        else VarNameSpace.noCondName
      }
      case Operator.Aggregate=>{
        if (node.cond.isDefined) {
          val layerName = LayerNameSpace.aggregatePredictEmbeddingLayerName
          val exprOutput = buildComputationGraphForSeqCond(layerName, relationMap, conf)
          exprOutput
        } else VarNameSpace.noCondName
      }
      case _ =>  VarNameSpace.noCondName
    }

    conf.addVertex(VarNameSpace.nodeEmbeddingInputName,new MergeVertex(),VarNameSpace.operatorName,outputVar)
    val columnEmbeddingOutputVar=insertMultipleLSTMLayerIntoComputationGraph(Seq(Seq(VarNameSpace.nodeEmbeddingInputName)),Seq(ModelDim.nodeEmbeddingDim),
      Seq(ModelDim.hiddenVectorDim),LayerNameSpace.nodeEmbeddingLayer,getModel(LayerNameSpace.nodeEmbeddingLayer),conf)

    val cardEstVar=insertMultipleLayerIntoComputationGraph(columnEmbeddingOutputVar,LayerNameSpace.cardEstLayer,
      getModel(LayerNameSpace.cardEstLayer),conf)
    conf.addLayer(VarNameSpace.finalEstimateResult,new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build(),cardEstVar)
    conf.setOutputs(VarNameSpace.finalEstimateResult)
    (new ComputationGraph(conf.build()),columnEmbeddingOutputVar,cardEstVar)
  }
  def buildComputationGraphForSeqCond(layerName:String,relationMap:mutable.Map[String,mutable.Set[String]],conf: => GraphBuilder):String={
    val inputAttributeSeq=relationMap.values.toSeq.reduce[mutable.Set[String]]((x,y)=> x++y).toSeq.map(x=>s"${VarNameSpace.columnEmbeddingPrefix}/${x}")
    insertMultipleLSTMLayerIntoComputationGraph(Seq(inputAttributeSeq),Seq(ModelDim.columnEmbeddingDim),Seq(ModelDim.predictEmbeddingDim),
      layerName,getModel(layerName),conf)

  }
  def buildComputationGraphForCond(cond:Cond,relations:mutable.Map[String,Relation],condMap:mutable.Map[Cond,String],conf: =>GraphBuilder,id:Int):(Int,String)=cond.getCondType match{
  case LogicCondType => {
    cond.condType match {
      case Condition.And => {
        val (leftId,leftOutput)=buildComputationGraphForCond(cond.children.get(0), relations, condMap,conf, id)       //返回最后输出的id(未使用）,返回的id为最小的可用的id号
        val (rightId,rightOutput)=buildComputationGraphForCond(cond.children.get(1),relations,condMap,conf, leftId)
        conf.addVertex(VarNameSpace.expPrefix + Integer.toString(rightId),new ScaleVertex(-1),leftOutput)
        conf.addVertex(VarNameSpace.expPrefix + Integer.toString(rightId+1),new ScaleVertex(-1),rightOutput)
        conf.addVertex(VarNameSpace.expPrefix + Integer.toString(rightId+2),new ElementWiseVertex(ElementWiseVertex.Op.Max),VarNameSpace.expPrefix+Integer.toString(rightId),VarNameSpace.expPrefix+Integer.toString(rightId+1))
        val exprOutput=VarNameSpace.expPrefix + Integer.toString(rightId+3)
        conf.addVertex(exprOutput,new ScaleVertex(-1),VarNameSpace.expPrefix + Integer.toString(rightId+2))
        (rightId+4,exprOutput)
      }
      case Condition.Or => {
        val (leftId,leftOutput)=buildComputationGraphForCond(cond.children.get(0),relations,condMap,conf,id)       //返回最后输出的id(未使用）,返回的id为最小的可用的id号
        val (rightId,rightOutput)=buildComputationGraphForCond(cond.children.get(1),relations,condMap,conf,leftId)
        val exprOutput=VarNameSpace.expPrefix + Integer.toString(rightId)
        conf.addVertex(exprOutput,new ElementWiseVertex(ElementWiseVertex.Op.Max),leftOutput,rightOutput)
        (rightId+1,exprOutput)
      }
      case Condition.Not => {
        val (childId,childOutput)=buildComputationGraphForCond(cond.children.get(0),relations,condMap,conf,id)       //childId即最小可使用的exprId,childOutput即子节点在计算图上返回的变量名
        val exprOutput=VarNameSpace.expPrefix + Integer.toString(childId)
        conf.addVertex(exprOutput,new ScaleVertex(-1),childOutput)
        (childId+1,exprOutput)
      }
      case _ => throw new Exception("Parse Failed!")
    }
  }
  case ArithmeticCondType=> {
    assert(condMap.contains(cond))
    (id,s"${VarNameSpace.atomicExpressionPrefix}/${condMap(cond)}")
  }
  case _=> throw  new Exception("parse error")
  }
}