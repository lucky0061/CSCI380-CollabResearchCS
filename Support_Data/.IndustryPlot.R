#industry compareation
IndPlot=function(performfile){
  algname=read.csv(AlgName,header = F)$V1
  filename=paste(".NewHS300_Performance_",algname,".csv",sep = "")
  N=length(algname)
  annret=NULL
  annsharpe=NULL
  annmaxdd=NULL
  win=NULL
  #data1=read.csv(filename[1],header = T)[-425,]
  data1=read.csv(filename[1],header = T)
  ind=read.csv(".industry.csv",header = T)
  spret=data1$AnnRetIndex
  byhret=data1$AnnRetBAH
  spsharpe=data1$AnnSharpeIndex
  byhsharpe=data1$AnnSharpeBAH
  spmaxdd=data1$MaxDDIndex
  byhmaxdd=data1$MaxDDBAH
  for(i in 1:N){
    win0=read.csv(filename[i],header = T)$Precision
    win=cbind(win,win0)
    traeret0=read.csv(filename[i],header = T)$AnnRetTrade
    annret=cbind(annret,traeret0)
    tradesharpe0=read.csv(filename[i],header = T)$AnnSharpeTrade
    annsharpe=cbind(annsharpe,tradesharpe0)
    trademaxdd0=read.csv(filename[i],header = T)$MaxDDTrade
    annmaxdd=cbind(annmaxdd,trademaxdd0)
  }
  #data2_1=data.frame(ind$Industry,spret[-425],byhret[-425],annret[-425,])
  #data2_2=data.frame(ind$Industry,spsharpe[-425],byhsharpe[-425],annsharpe[-425,])
  #data2_3=data.frame(ind$Industry,spmaxdd[-425],byhmaxdd[-425],annmaxdd[-425,])
  #data2_4=data.frame(ind$Industry,win[-425,])
  data2_1=data.frame(ind$Industry,spret,byhret,annret)
  data2_2=data.frame(ind$Industry,spsharpe,byhsharpe,annsharpe)
  data2_3=data.frame(ind$Industry,spmaxdd,byhmaxdd,annmaxdd)
  data2_4=data.frame(ind$Industry,win)
  colnames(data2_1)=c("Industry","HS300 Index","Buy and Hold",as.character(algname))
  colnames(data2_2)=c("Industry","HS300 Index","Buy and Hold",as.character(algname))
  colnames(data2_3)=c("Industry","HS300 Index","Buy and Hold",as.character(algname))
  colnames(data2_4)=c("Industry",as.character(algname))
  M=15
  meanwin=NULL
  meanret=NULL
  meansharp=NULL
  meanmaxdd=NULL
  for(i in 2:M){
    a1=data2_1[,c(1,i)]
    a2=data2_2[,c(1,i)]
    a3=data2_3[,c(1,i)]
    meanret0=tapply(a1[,2], INDEX =a1[,1],FUN = mean)
    meansharp0=tapply(a2[,2], INDEX =a2[,1], FUN =mean)
    meanmaxdd0=tapply(a3[,2], INDEX =a3[,1], FUN =mean)
    meanret=cbind(meanret,meanret0)
    meansharp=cbind(meansharp,meansharp0)
    meanmaxdd=cbind(meanmaxdd,meanmaxdd0)
  }
  meanwin=NULL
  for (i in 2:13) {
    a4=data2_4[,c(1,i)]
    meanwin0=tapply(a4[,2], INDEX =a4[,1],FUN = mean)
    meanwin=cbind(meanwin,meanwin0)
  }
  data3_1=data.frame(unname(meanret),rownames(meanret))
  data3_2=data.frame(unname(meansharp),rownames(meansharp))
  data3_3=data.frame(unname(meanmaxdd),rownames(meanmaxdd))
  data3_4=data.frame(unname(meanwin),rownames(meanwin))
  colnames(data3_1)=c("StockIndex","Buy and Hold",as.character(algname),"Industry")
  colnames(data3_2)=c("StockIndex","Buy and Hold",as.character(algname),"Industry")
  colnames(data3_3)=c("StockIndex","Buy and Hold",as.character(algname),"Industry")
  colnames(data3_4)=c(as.character(algname),"Industry")
  data4_1=reshape2::melt(data3_1,id.vars="Industry")
  data4_2=reshape2::melt(data3_2,id.vars="Industry")
  data4_3=reshape2::melt(data3_3,id.vars="Industry")
  data4_4=reshape2::melt(data3_4,id.vars="Industry")
  #colnames(data4_1)=c("Industry","TradingAlgorithm"," AnnualizedReturn")
  library(ggplot2)
  p1=ggplot(data4_1,aes(x=Industry,y= value,fill=variable))+geom_bar(position="dodge",stat="identity")+xlab("Industry")+ylab("Annualized Return")+labs("Trading Algorithm")+ggplot2::theme(axis.text.x=ggplot2::element_text(color="black",size = 8),axis.text.y=ggplot2::element_text(color="black",size = 8))
  p2=ggplot(data4_2,aes(x=Industry,y= value,fill=variable))+geom_bar(position="dodge",stat="identity")+xlab("Industry")+ylab("Annualized Sharpe Rate")+labs("Trading Algorithm")+ggplot2::theme(axis.text.x=ggplot2::element_text(color="black",size = 8),axis.text.y=ggplot2::element_text(color="black",size = 8))
  p3=ggplot(data4_3,aes(x=Industry,y= -value,fill=variable))+geom_bar(position="dodge",stat="identity")+xlab("Industry")+ylab("Maximum Drowdown")+labs("Trading Algorithm")+ggplot2::theme(axis.text.x=ggplot2::element_text(color="black",size = 8),axis.text.y=ggplot2::element_text(color="black",size = 8))
  return(list(p1,p2,p3))
  }
IndPlot(".BoxPlot.csv")
#----x=data2_1[,c(1:9)]--------------------------------------------------------------
x=data2_4[,c(1:13)]
x_1=x[which(x$Industry=="EN"),]
x_11=reshape2::melt(x_1)
round(tapply(x_11$value,x_11$variable,mean),4)
max(round(tapply(x_11$value,x_11$variable,mean),4))
bartlett.test(x_11$value~x_11$variable,data = x_11)
kruskal.test(x_11$value~x_11$variable,data = x_11)
pgirmess::kruskalmc(x_11$value~x_11$variable,data = x_11)
#agricolae::kruskal(x_11$value, "x_11$variable)
a=DescTools::NemenyiTest(x_11$value~x_11$variable,data = x_11)
write.csv(a[[1]],file=".MDD_industry.csv")
wilcox.test(x_11$value~x_11$variable,data = x_11)
#fligner.test(x11$value, x11$variable)
x_111=aov(x_11$value~x_11$variable,data = x_11)
x_1111=TukeyHSD(x_111)
x_1111
x_1111$`x11$variable`[,4][which(x_1111$`x11$variable`[,4]<0.05)]