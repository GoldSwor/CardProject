package org.example

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.hadoop.io.IOUtils
import org.apache.spark.sql.catalyst.catalog.HiveTableRelation
import org.apache.spark.sql.catalyst.expressions.{And, AttributeReference, Cast, EqualTo, Expression, GreaterThan, GreaterThanOrEqual, In, IsNotNull, IsNull, LessThan, LessThanOrEqual, Literal, NamedExpression, Not, Or, StringRegexExpression}
import org.apache.spark.sql.catalyst.plans.{Cross, FullOuter, Inner, JoinType, LeftAnti, LeftOuter, LeftSemi, RightOuter}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, GlobalLimit, Join, LeafNode, LocalLimit, LocalRelation, LogicalPlan, Project, Union}
import org.apache.spark.sql.types.LongType
import org.json4s.jackson.Json

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util.Date
import scala.util.matching.UnanchoredRegex


object PlanUtils{
  val PATTERN_MISMATCHING_ERROR="Pattern Mismatching"
  val labelPattern="^\\[(\\d\\s+,)*(\\d+)\\]$".r
  var tree=Tree()
  def serializeLogicalPlan(plan:LogicalPlan):Array[Byte]={
    val tree=transformLogicalPlanToTree(plan)
    val out=new ByteArrayOutputStream()
    val objectOut=new ObjectOutputStream(out)
    objectOut.writeObject(tree)
    objectOut.close()
    out.toByteArray
  }
  def deserializeLogicalPlan(planByte:Array[Byte]):Tree={
    val in=new ObjectInputStream(new ByteArrayInputStream(planByte))
    val tree=in.readObject().asInstanceOf[Tree]
    in.close()
    tree
  }
  def persistPlanToFile(fs:FileSystem,planPath:String,plan:LogicalPlan)={
    val time=new Date().getTime
    val path=s"${planPath}/${time}.ser"
    val outputStream=fs.create(new Path(path))
    outputStream.write(serializeLogicalPlan(plan))
    outputStream.close()
  }

  /**
   *
   * @param uri
   * @param planPath
   * @param user
   * @param timeLine
   * @param count
   * @return 1、返回时间戳在timeLine之后的count个查询计划树
   *         2、训练的文件名
   */
  def readPlanTreeFromFile(fs:FileSystem,planPath:String,timeLine:Long,count:Int):Seq[(Tree,String)]={
    FileUtil.stat2Paths(fs.listStatus(new Path(s"${planPath}"))
      .filter(x=>x.isFile))
      .filter(x=>
        (getTimeLineFromFileName(x.getName)>=timeLine)).zipWithIndex
      .filter(x=>x._2<count).map(x=>x._1)
      .map(f=> {
        val in=fs.open(f)
        val tree=new ObjectInputStream(in).readObject().asInstanceOf[Tree]
        in.close()
        (tree,f.getName)
      }).toSeq
  }
  def parseArrayString(arrayString:String):Seq[Long]={
     assert(labelPattern.findFirstIn(arrayString).isDefined)
        arrayString
          .replaceFirst("\\[","")
          .replaceFirst("\\]","")
          .split(",")
          .map(x=>x.toLong)
  }
  def readLabelsFromFile(fs:FileSystem,labelsPath:String*):Seq[Seq[Long]]={
      labelsPath.map(x=>{
        val in=fs.open(new Path(x))
        val arrayString=in.readUTF()
        in.close()
        parseArrayString(arrayString)
      })
  }
  def getTimeLineFromFileName(path:String):Long={
    val pattern="(\\.\\w+)?$".r
    pattern.replaceFirstIn(path.split("//").last.split("/").last,"").toLong
  }
  //just transform valid pattern in PlanTree
  def transformSubLogicalPlanToTree(root:LogicalPlan):Seq[Tree]={
    try{
      val ans=root match{
        case q:LeafNode=>Seq()
        case _ => Seq(transformLogicalPlanToTree(root))
      }
      ans
    }catch {
      case _:Exception=>{
        root.children.foldLeft(List[Tree]())((x,y)=>x++transformSubLogicalPlanToTree(y).toList)
      }
    }
  }
  def transformLogicalPlanToTree(plan:LogicalPlan):Tree={
    val root=transformLogicalPlanToPlanNode(plan) match{
      case p:PlanNode=>p
      case _ => throw new Exception("Root Node Can Not Be One Relation Node")
    }
    println(plan.toString())
    tree.setRoot(root)
    val ans=tree
    tree=Tree()
    ans
  }
  def transformLogicalPlanToPlanNode(plan:LogicalPlan):TreeNode=plan match{
    case Join(left, right, joinType, condition, _) => {
      val children = transformLogicalPlanToPlanNode(left) :: transformLogicalPlanToPlanNode(right) :: Nil
      val cond: Option[Seq[Cond]] = {
        if (condition.isDefined) {
          Some((transformExpressionToCond(condition.get) :: Nil).toSeq)
        }
        else {
          None
        }
      }
      new PlanNode(getJoinType(joinType), cond, Some(children.toSeq))
    }
    case Filter(cond, child) => {
      val children = transformLogicalPlanToPlanNode(child) :: Nil
      new PlanNode(Operator.Filter, Some(Seq(transformExpressionToCond(cond))), Some(children.toSeq))
    }
    case Aggregate(groupingExpressions, _, child) => {
      val cond = {
        if (groupingExpressions.isEmpty) {
          None
        }
        else {
          Some(groupingExpressions.map(e => transformExpressionToCond(e)).toSeq)
        }
      }
      new PlanNode(Operator.Aggregate, cond, Some(Seq(transformLogicalPlanToPlanNode(child))))
    }
    case Union(children, _, _) => new PlanNode(Operator.Union, None, Some(children.map(x => transformLogicalPlanToPlanNode(x))))
    case Project(projectList, child) => {
      new PlanNode(Operator.Project, Some(projectList.map(e => transformExpressionToCond(e))), Some(Seq(transformLogicalPlanToPlanNode(child))))
    }
    case l: LocalLimit => {
      val cond = {
        if (l.maxRowsPerPartition.isDefined) {
          val literal = new Literal(l.maxRowsPerPartition.get, LongType)
          Some(Seq(new Cond(Condition.Literal, None, Some(literal), None)))
        }
        else None
      }
      new PlanNode(Operator.LocalLimit, cond, Some(Seq(transformLogicalPlanToPlanNode(l.child))))
    }
    case g: GlobalLimit => {
      val cond = {
        if (g.maxRows.isDefined) {
          val literal = new Literal(g.maxRows.get, LongType)
          Some(Seq(new Cond(Condition.Literal, None, Some(literal), None)))
        }
        else None
      }
      new PlanNode(Operator.GlobalLimit, cond, Some(Seq(transformLogicalPlanToPlanNode(g.child))))
    }
    //    case Intersect(left, right, isAll) => {
    //      val children = Some(Seq(left, right).map(p => transformLogicalPlanToPlanNode(p)))
    //      if (isAll) {
    //        new PlanNode(Operator.IntersectAll, None, children)
    //      }
    //      else {
    //        new PlanNode(Operator.Intersect, None, children)
    //      }
    //    }
    //    case Except(left, right, isAll) => {
    //      val children = Some(Seq(left, right).map(p => transformLogicalPlanToPlanNode(p)))
    //      if (isAll) {
    //        new PlanNode(Operator.ExceptAll, None, children)
    //      }
    //      else {
    //        new PlanNode(Operator.Except, None, children)
    //      }
    //    }
    case p: LeafNode => {
      val ans = p match {
        case t: HiveTableRelation => {
          val rel = new Relation("HiveTableRelation", t.tableMeta.qualifiedName, t.schema, t.computeStats())
          tree.addRelation(rel) //side effect happening
          rel
        }
        case l:LocalRelation =>{
          new Relation("LocalRelation","ss",l.schema,l.computeStats())
        }
        case _ => throw new Exception(PATTERN_MISMATCHING_ERROR)
      }
      ans
    }
    case p: LogicalPlan => {
      new PlanNode(Operator.Null, None, Some(p.children.map(child => transformLogicalPlanToPlanNode(child))))
    }
  }
  def getJoinType(joinType:JoinType):Operator.Operator=joinType match{
    case Inner=>Operator.Inner
    case FullOuter => Operator.FullOuter
    case LeftOuter => Operator.LeftOuter
    case RightOuter => Operator.RightOuter
    case LeftSemi => Operator.LeftSemi
    case LeftAnti => Operator.LeftAnti
    case Cross => Operator.Cross
    case _ => throw new Exception(PATTERN_MISMATCHING_ERROR)
  }
  def transformExpressionToCond(expression: Expression):Cond=expression match{
    case And(left,right)=>{
      val children=transformExpressionToCond(left)::transformExpressionToCond(right)::Nil
      new Cond(Condition.And,Some(children.toSeq),None,None)
    }
    case Or(left,right)=>{
      val children=transformExpressionToCond(left)::transformExpressionToCond(right)::Nil
      new Cond(Condition.Or,Some(children.toSeq),None,None)
    }
    case Not(child) =>{
      val children=transformExpressionToCond(child)::Nil
      new Cond(Condition.Not,Some(children.toSeq),None,None)
    }
    case s:StringRegexExpression => {
      val children=transformExpressionToCond(s.left)::transformExpressionToCond(s.right)::Nil
      new Cond(Condition.Like,Some(children.toSeq),None,None)
    }
    case Cast(child,_,_)=>{
      transformExpressionToCond(child)
    }
    case IsNull(child) =>{
      new Cond(Condition.IsNull,Some(List(transformExpressionToCond(child)).toSeq),None,None)
    }
    case IsNotNull(child) => {
      new Cond(Condition.IsNotNull,Some(List(transformExpressionToCond(child)).toSeq),None,None)
    }
    case EqualTo(left,right) =>{
      val children=transformExpressionToCond(left)::transformExpressionToCond(right)::Nil
      new Cond(Condition.Equal,Some(children.toSeq),None,None)
    }
    case GreaterThan(left,right) =>{
      val children=transformExpressionToCond(left)::transformExpressionToCond(right)::Nil
      new Cond(Condition.Greater,Some(children.toSeq),None,None)
    }
    case GreaterThanOrEqual(left,right)=>{
      val children=transformExpressionToCond(left)::transformExpressionToCond(right)::Nil
      new Cond(Condition.GreaterEqual,Some(children.toSeq),None,None)
    }
    case LessThan(left,right) =>{
      val children=transformExpressionToCond(left)::transformExpressionToCond(right)::Nil
      new Cond(Condition.Less,Some(children.toSeq),None,None)
    }
    case LessThanOrEqual(left,right) =>{
      val children=transformExpressionToCond(left)::transformExpressionToCond(right)::Nil
      new Cond(Condition.LessEqual,Some(children.toSeq),None,None)
    }
    case In(child,list) =>{
      //transform in clause to or clause,eg: a in [1,2,3] is equal to a=1 or a=2 or a=3
      val f=(x:Expression,y:Expression)=>new Cond(Condition.Equal,Some(List(transformExpressionToCond(x),transformExpressionToCond(y)).toSeq),None,None)
      val init=f(child,list(0))
      list.foldLeft(init)((x,y)=>new Cond(Condition.Or,Some(List(x,f(child,y)).toSeq),None,None))
    }
    case l:Literal=>{
      new Cond(Condition.Literal,None,Some(l),None)
    }
    case n:NamedExpression=>n match{
      case n0:AttributeReference=> new Cond(Condition.Attribute,None,None,Some(n0.qualifiedName),Some(n0.exprId.id))
      case _=>throw new Exception(PATTERN_MISMATCHING_ERROR)

    }
    case _ => throw new Exception(PATTERN_MISMATCHING_ERROR)
  }
}
