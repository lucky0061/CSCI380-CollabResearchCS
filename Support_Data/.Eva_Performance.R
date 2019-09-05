# evaluate function and significant test
backtest=function(ret,tradeRet){
  performance=function(x){
    winpct=length(x[x>0])/length(x[x!=0])
    annRet=PerformanceAnalytics::Return.annualized(x,scale = 250,geometric = F)
    sharpe=PerformanceAnalytics::SharpeRatio.annualized(x,scale = 250,geometric = F)
    maxDD=PerformanceAnalytics::SharpeRatio.annualized(x,scale = 250,geometric = F)
    perfo=c(winpct,annRet,sharpe,maxDD)
    return(perfo)
  }
  cbind(performance(ret),performance(tradeRet))
}
#-------------------------Performance Function---------------------
eva_func=function(sec_names){
  idNumber=read.csv(sec_names,header=F)$V1
  Stock_Name=read.csv(sec_names,header=F)$V2
  #idNumber=substr(as.character(idNumber),start = 1,stop = 6)#".NewHS300.csv"
  Label_File=paste(idNumber,"_DatasetNew",".csv",sep = "")
  SignalFile=paste(idNumber,"_TradeSignal_RNN_Lasso",".csv",sep = "")
  Rawfile=paste(idNumber,"_New",".csv",sep = "")
  A_Index=read.csv(".SP500_Index.csv",header = T)
  #A_Index=read.csv(".HS300_Index.csv",header = T)
  A_Index=xts::xts(A_Index[,2:6],order.by=as.Date(A_Index$X))
  N=length(SignalFile)
  WinIndex=NULL
  WinBAH=NULL
  WinTrade=NULL
  AnnRetIndex=NULL
  AnnRetBAH=NULL
  AnnRetTrade=NULL
  AnnSharpeIndex=NULL
  AnnSharpeBAH=NULL
  AnnSharpeTrade=NULL
  MaxDDIndex=NULL
  MaxDDBAH=NULL
  MaxDDTrade=NULL
  Accuracy=NULL
  Precision=NULL
  Recall=NULL
  for(i in 1:N){
    TradeSignal0=read.csv(SignalFile[i],header = T)
    TradeSignal=xts::xts(TradeSignal0[,2],order.by = as.Date(TradeSignal0[,1]))
    TradeSignal=lag(TradeSignal,1)
    RawPrice=read.csv(Rawfile[i],header = T)
    RawPrice=xts::xts(RawPrice[,-1],order.by = as.Date(RawPrice[,1]))
    RawRet=quantmod::dailyReturn(RawPrice[,"Close"])[zoo::index(TradeSignal)]
    SpRet=quantmod::dailyReturn(A_Index[,"Close"])[zoo::index(TradeSignal)]
    ReturnTrade=TradeSignal[-1]*RawRet[-1]
    EvalIndi1=backtest(SpRet,ReturnTrade)
    EvalIndi2=backtest(RawRet,ReturnTrade)
    WinIndex0=EvalIndi1[1,1]
    WinBAH0=EvalIndi2[1,1]
    WinTrade0=EvalIndi1[1,2]
    AnnRetIndex0=EvalIndi1[2,1]
    AnnRetBAH0=EvalIndi2[2,1]
    AnnRetTrade0=EvalIndi1[2,2]
    AnnSharpeIndex0=EvalIndi1[3,1]
    AnnSharpeBAH0=EvalIndi2[3,1]
    AnnSharpeTrade0=EvalIndi1[3,2]
    MaxDDIndex0=EvalIndi1[4,1]
    MaxDDBAH0=EvalIndi2[4,1]
    MaxDDTrade0=EvalIndi1[4,2]
    WinIndex=c(WinIndex,WinIndex0)
    WinBAH=c(WinBAH,WinBAH0)
    WinTrade=c(WinTrade,WinTrade0)
    AnnRetIndex=c(AnnRetIndex,AnnRetIndex0)
    AnnRetBAH=c(AnnRetBAH,AnnRetBAH0)
    AnnRetTrade=c(AnnRetTrade, AnnRetTrade0)
    AnnSharpeIndex=c(AnnSharpeIndex, AnnSharpeIndex0)
    AnnSharpeBAH=c(AnnSharpeBAH,AnnSharpeBAH0)
    AnnSharpeTrade=c(AnnSharpeTrade,AnnSharpeTrade0)
    MaxDDIndex=c(MaxDDIndex,MaxDDIndex0)
    MaxDDBAH=c(MaxDDBAH,MaxDDBAH0)
    MaxDDTrade=c(MaxDDTrade,MaxDDTrade0)
    Label=read.csv(Label_File[i],header = T)$Label[-c(1:250)]
    Signal_Label=TradeSignal0[,2][-c(1:250)]
    result=table(Signal_Label,Label)
    if(dim(result)[1]==2){
      Accuracy0=sum(diag(result))/sum(result)
      Precision0=result[2,2]/sum(result[,2])
      Recall0=result[2,2]/sum(result[2,])
    }else{
      Accuracy0=1
      Precision0=1
      Recall0=1
    }
    Accuracy=c(Accuracy,Accuracy0)
    Precision=c(Precision,Precision0)
    Recall=c(Recall,Recall0)
    }
    performance=cbind(WinIndex,WinBAH,WinTrade,AnnRetIndex,AnnRetBAH,AnnRetTrade,AnnSharpeIndex,AnnSharpeBAH,AnnSharpeTrade,MaxDDIndex,MaxDDBAH,MaxDDTrade,Accuracy,Precision,Recall)
    rownames(performance)=idNumber
    colnames(performance)=c("WinIndex","WinBAH","WinTrade","AnnRetIndex","AnnRetBAH","AnnRetTrade","AnnSharpeIndex","AnnSharpeBAH","AnnSharpeTrade","MaxDDIndex","MaxDDBAH","MaxDDTrade","Accuracy","Precision","Recall")
    write.csv(as.data.frame(performance),file=".NewSP500_Performance_RNN_Lasso.csv")
     }
eva_func(".NewSP500.csv")#".NewHS300.csv"