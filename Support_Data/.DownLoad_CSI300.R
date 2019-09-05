#download A stocks data from wangyi finance
#".NewHS300.csv"
Download_A=function(sec_name){
  stock_names=read.csv(sec_name,header = F)$V1
  N=length(stock_names)
  base_url="http://quotes.money.163.com/service/chddata.html?code=1"
  end_url="&start=19991110&end=20181229&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP"
  for(i in 1:N){
    url=paste(base_url,substr(as.character(stock_names[i]),start = 1,stop = 6),end_url,sep = "")
    stock_data=read.csv(url)
    stock_data=stock_data[,c(1,4,5,6,7,12)]
    colnames(stock_data)=c("Date","Close","High","Low","Open","Volume")
    stock_data=xts::xts(stock_data[,-1],order.by = as.Date(stock_data$Date))
    stock_data=stock_data[stock_data$Volume!=0,]
    newdata=paste(substr(as.character(stock_names[i]),start = 1,stop = 6),".csv",sep = "")
    #newdata=paste(stock_names[i],".csv",sep = "")
    write.csv(as.data.frame(stock_data),file = newdata)
  }
}
Download_A(".NewHS300.csv")