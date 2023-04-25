package org.example

import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.catalyst.plans.logical.statsEstimation.EstimationUtils
import org.apache.spark.sql.types.{BinaryType, ByteType, DoubleType, FloatType, IntegerType, LongType, ShortType, StringType}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.example.ArithmeticCond.ArithmeticCond
import org.example.Operator.Operator

import scala.util.hashing.MurmurHash3
object NodeEncode {
  val seed=123
  val hashLen=128
  val EPS=1e-30
  //所有编码的rank均为2，shape 为 [1,codeLen]
  case class EncodeErrorException(val msg:String) extends Exception{
    def info:String=this.msg
  }
  def encodeString(s:String):INDArray={
    val code=Nd4j.zeros(1,hashLen)
    s.map(x=>MurmurHash3.stringHash(x.toString,seed)%hashLen).foreach(id=>code.putScalar(id,1.0))
    code
  }
  def onehotEncode(idx:Int,dim:Int):INDArray={
    Nd4j.zeros(1,dim).putScalar(Array[Int](0,idx),1.0)
  }
  def encodeOperator(node:PlanNode):INDArray={
    onehotEncode(node.getOperatorIdx,Operator.Dim)
  }
  def encodeArithmeticCond(cond:Cond):INDArray={
    if(!cond.getCondType.equals(ArithmeticCondType)) throw EncodeErrorException(s"cond ${cond.getCondType} is not arithmetic cond")
    onehotEncode(cond.getCondIdx,ArithmeticCond.Dim)
  }
  def encodeColumn(cond: Cond,relation: Relation):INDArray={
    if(!cond.getCondType.equals(LeafCondType)) throw EncodeErrorException(s"cond ${cond.getCondType} is not attribute cond")
    if(!cond.condType.equals(Condition.Attribute)) throw  EncodeErrorException(s"cond ${cond.getCondType} is not attribute cond")
    val columnName=cond.name.get.split('.').last
    val idx=relation.getIdx(columnName)
    onehotEncode(idx,relation.getColumnNum)
  }
  def encodeLiteral(cond:Cond,relation:Relation):INDArray={
    if(!cond.getCondType.equals(LeafCondType)) throw EncodeErrorException(s"cond ${cond.getCondType} is not attribute cond")
    if(!cond.condType.equals(LeafCond.Literal)) throw  EncodeErrorException(s"cond ${cond.getCondType} is not attribute cond")
    val literal=cond.value.get
    val columnStat=relation.stat.attributeStats.map(x=>(x._1.qualifiedName->x._2)).get(cond.name.get).get //get corresponding columnStat
    literal.dataType match{
      case StringType => {
        encodeString(literal.value.asInstanceOf[String])
      }
      case BinaryType =>{
        encodeString(literal.value.asInstanceOf[Array[Byte]].map(b=>b.toChar).mkString)
      }
      case _ if columnStat.min.isEmpty || columnStat.max.isEmpty => throw EncodeErrorException(s"${cond.name} min-max statistic does not exists")
      case _ => {
        val min=EstimationUtils.toDouble(columnStat.min.get,literal.dataType)
        val max=EstimationUtils.toDouble(columnStat.max.get,literal.dataType)
        val value=(EstimationUtils.toDouble(literal.value,literal.dataType)-min)/(max-min+EPS)
        Nd4j.zeros(1,hashLen).putScalar(0,value)
      }
    }
  }
}

