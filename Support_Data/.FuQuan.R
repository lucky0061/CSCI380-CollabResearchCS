# CSI300/hs300 ex-weight/right
fuquan=function(sec_name){
  stock_names=read.csv(sec_name,header = F)$V1
  stock_names=substr(as.character(stock_names),start = 1,stop = 6)
  N=length(stock_names)
  for(i in 1:N){
    Stock_File=paste(stock_names[i],".csv",sep = "")
    NewFile=paste(stock_names[i],"_New",".csv",sep = "")
    Stock_Data=read.csv(Stock_File,header = T)
    Stock_Data=xts::xts(Stock_Data[,-1],order.by = as.Date(Stock_Data[,1]))["2004-01-01/2018-12-29"]
    DS_File=paste(stock_names[i],"_DS",".csv",sep="")
    DS_Data=read.csv(DS_File,header = T)
    DS_Data=xts::xts(DS_Data[,-1],order.by = as.Date(DS_Data[,1]))["2004-01-01/2018-12-29"]
    M=dim(DS_Data)[1]
    Date=zoo::index(DS_Data) 
    s_data=Stock_Data[,1:4]
    for(j in 1:M){
      date=paste("/",Date[j]-1,sep = "")
      date1=paste(Date[j],"/",sep = "")
      s_data0=s_data[date]
      s_data1=zoo::coredata(s_data0)
      DS_Data1=zoo::coredata(DS_Data)
      s_data=apply(s_data1,2,function(x){return((x-DS_Data1[j,3]/10)/(1+DS_Data1[j,1]/10+DS_Data1[j,2]/10))})
      s_data=xts::xts(s_data,order.by = zoo::index(s_data0))
      Stock_Data1=xts::rbind.xts(s_data,Stock_Data[date1,1:4])
      s_data=Stock_Data1
    }
    Stock_Data=xts::cbind.xts(s_data,Stock_Data[,5])
    write.csv(as.data.frame(Stock_Data),file=NewFile)
  }
}
fuquan(".NewHS300.csv")