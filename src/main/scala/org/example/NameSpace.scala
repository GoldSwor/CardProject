package org.example

object VarNameSpace{
  val atomicExpressionPrefix="atomic_expression_embedding"
  val columnEmbeddingPrefix="column_embedding"
  val columnEmbeddingPreprocessPrefix="column_embedding_preprocess"
  val cardEstPrefix="card_estimate"
  val yLastOutputName="node_embedding_layer_output"
  val noCondName="no_cond_encode"
  val operatorName=s"${cardEstPrefix}/operator"
  val yLastInputName=s"${cardEstPrefix}/yLast"
  val nodeEmbeddingInputName="node_embedding_layer_input"
  val expPrefix="exp"
  val condPrefix="cond"
  val zeroEncoding="constant0"
  val finalEstimateResult="finalCardResult"
}
object LayerNameSpace{
  val aggregatePredictEmbeddingLayerName = "aggregate_predict_embedding_layers" //aggregate算子的predict层,LSTM层
  val filterPredictEmbeddingLayerName = "filter_predict_embedding_layers" //filter算子的predict层
  val joinPredictEmbeddingLayerName = "join_predict_embedding_layers" //join算子的predict层
  val cardEstLayer="card_est_layers" //基数估计层
  val nodeEmbeddingLayer="node_embedding_layers" //Tree LSTM层
}
object ModelDim{
  val predictEmbeddingDim=128 // atomic predict embedding layer输出维度
  val hiddenVectorDim=128 // TreeLstm 输出维度
  val columnEmbeddingDim=128 // column embedding model 输出维度
  val nodeEmbeddingDim=Operator.Dim+predictEmbeddingDim // PlanNode表示维度
  val predictDim=math.max(2*columnEmbeddingDim+ArithmeticCond.Dim,columnEmbeddingDim+ArithmeticCond.Dim+NodeEncode.hashLen) // atomic predict embedding layer输入维度
  val TLSTMInputDim=nodeEmbeddingDim+hiddenVectorDim // TreeLstm 输入维度
}
object LayerNum{
  val relationLayerNum=2
  val aggregatePredictEmbeddingLayerNum=1
  val filterPredictEmbeddingLayerNum=1
  val joinPredictEmbeddingLayerNum=1
  val cardEstLayerNum=2
  val nodeEmbeddingLayerNum=1
}
object LayerKind extends Enumeration{
  type LayerKind=Value
  val MLP,LSTM=Value
}
