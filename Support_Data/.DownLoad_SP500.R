#download S&P 500 stock data from Yahoo Finance.com
DownLoad=function(sec_names){
  idNumber=read.csv(sec_names,header=T)$Code
  idNumber=as.character(idNumber)
  outfiles=paste(idNumber,".csv",sep = "")
  N=length(idNumber)
  for(j in 1:N){
    #stock=timeSeries::as.timeSeries(quantmod::getSymbols(idNumber[j],from="2001-01-01",to="2018-02-15"))
    getSymbols(idNumber[j],from="2001-01-01",to="2018-02-15")
    write.csv(as.data.frame(eval(parse(text = idNumber[j]))),file = outfiles[j])
  }
}
DownLoad(".NewSP500.csv")
# Stock Price adjusted for Divided
AdjPrice=function(sec_names){
  idNumber=read.csv(sec_names,header=T)$Code
  idNumber=as.character(idNumber)
  rawfile=paste(idNumber,".csv",sep = "")
  outfiles=paste(idNumber,"_New",".csv",sep = "")
  N=length(idNumber)
  for(i in 1:N){
    stockdata=read.csv(rawfile[i],header = T)
    step=stockdata[,7]-stockdata[5]
    Open=stockdata[,2]+step
    High=stockdata[,3]+step
    Low=stockdata[,4]+step
    Close=stockdata[,5]+step
    Volume=stockdata[,6]
    stockdata1=xts::xts(data.frame(Open,High,Low,Close,Volume),order.by = as.Date(stockdata[,1]))
    colnames(stockdata1)=c("Open","High","Low","Close","Volume")
    stockdata2=na.omit(stockdata1)
    write.csv(as.data.frame(stockdata2),file=outfiles[i])
    }
}
AdjPrice(".NewSP500.csv")