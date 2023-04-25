package org.example

import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.types.StructType

abstract class TreeNode extends Serializable
import scala.collection.mutable
object Operator extends Enumeration with Serializable {
  type Operator=Value
  val Inner,FullOuter,LeftOuter,RightOuter,LeftSemi,LeftAnti,Cross,Filter,Aggregate,Union,Project,LocalLimit,GlobalLimit,Null=Value
  val Dim=14
}
sealed trait CondType extends Serializable
case object ArithmeticCondType extends CondType
case object LogicCondType extends CondType
case object LeafCondType extends CondType
object ArithmeticCond extends Enumeration with Serializable {
  type ArithmeticCond=Value
  val Equal,Greater,GreaterEqual,Less,LessEqual,Like,IsNull,IsNotNull=Value
  val Dim=8
}
object LogicCond extends Enumeration with Serializable{
  type LogicCond=Value
  val And,Or,Not=Value
}
object LeafCond extends Enumeration with Serializable {
  type LeafCond=Value
  val Literal,Attribute=Value
}
object Condition extends Enumeration with Serializable{
  type Condition=Value
  val Equal,Greater,GreaterEqual,Less,LessEqual,Like,IsNull,IsNotNull,And,Or,Not,Literal,Attribute=Value
}
class PlanNode(val nodeType:Operator.Operator,val cond: Option[Seq[Cond]],val children:Option[Seq[TreeNode]]) extends TreeNode with Serializable{
  def traverse:Unit={
    println(nodeType)
    for(child<-children.get){
      if(child.isInstanceOf[Relation]) println(child.asInstanceOf[Relation].relationType)
      else child.asInstanceOf[PlanNode].traverse
    }
  }
  def getOperatorIdx:Int={
    this.nodeType.id
  }
  def getOperator:Operator.Operator={
    this.nodeType
  }
}
class Cond(val condType:Condition.Condition,val children: Option[Seq[Cond]],val value:Option[Literal],val name:Option[String],val exprId:Option[Long]=None) extends Serializable{
  def getCondType:CondType=condType match{
    case c if c.id<ArithmeticCond.Dim => ArithmeticCondType
    case c if c.id>=ArithmeticCond.Dim && c.id<=ArithmeticCond.Dim+2 => LogicCondType
    case c if c.id>ArithmeticCond.Dim+2 => LeafCondType
  }
  def getCondIdx:Int=getCondType match{
    case ArithmeticCondType => this.condType.id
    case LogicCondType => this.condType.id-ArithmeticCond.Dim
    case LeafCondType => this.condType.id-(ArithmeticCond.Dim+3)
  }
  def getRelationForCond(relations:mutable.Map[String,Relation]):Relation=this.condType match{
    case Condition.Attribute => {
      if(this.name.isDefined){
        val qualifier=this.name.get.split(".")//qualifier: database.table.column
        assert(qualifier.length==3)
        val database=qualifier(0)
        val table=qualifier(1)
        val relation=relations.get(s"${database}.${table}")
        assert(relation.isDefined)
        relation.get
      }
      else throw new Exception("attribute cond has no name!")
    }
    case _ => throw new Exception(s"${this.condType} has no attribute name!")
  }
}
class Relation(val relationType:String,val name:String,val schema:StructType,val stat:Statistics) extends TreeNode with Serializable{
  def getColumnNum:Int=this.schema.length
  def getIdx(column:String):Int={
    val columnSeq=this.schema.map(f=>f.name)
    assert(columnSeq.exists(column => true))
    columnSeq.indexOf(column)
  }
}
object RelationExistException extends Exception
class Tree(var root:Option[PlanNode],var relations:mutable.Map[String,Relation]) extends Serializable {
  //a container which store the tree's temporary storage
  def setRoot(root:PlanNode)={
    this.root=Some(root)
  }
  def addRelation(relation:Relation)={
    if(this.relations.get(relation.name).isDefined) throw RelationExistException
    else this.relations(relation.name)=relation
  }
  def addRelations(relations: Relation*)={
    for(relation <- relations){
      this.addRelation(relation)
    }
  }
  def setRelation(relations: mutable.Map[String,Relation])={
    this.relations=relations
  }
  def getRelation={
    this.relations
  }
  def clearAll={
    //also can be used as an initializer or clear information stored in the tree
    this.root=None
    this.setRelation(mutable.Map[String,Relation]())
  }

}
object Tree{
  //construct one tree without root node and with empty relation map
  def apply(): Tree = new Tree(None,mutable.Map[String,Relation]())
}




