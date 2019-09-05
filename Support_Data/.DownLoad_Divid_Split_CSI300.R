#Download Divid and Split Information of China A Stock from wangyi finance!
#".NewHS300.csv"
Divid_Split=function(sec_name){
  stock_names=read.csv(sec_name,header = F)$V1
  N=length(stock_names)
  base_url="http://quotes.money.163.com/f10/fhpg_"
  end_url=".html#01d05"
  for(i in 1:N){
    url=paste(base_url,substr(as.character(stock_names[i]),start = 1,stop = 6),end_url,sep = "")
    #url=paste(base_url,stock_names[i],end_url,sep = "")
    source=readLines(url,encoding = "UTF-8")
    DS_data=XML::readHTMLTable(source)[[4]]
    M=dim(DS_data)[1]
    DS_data=DS_data[3:M,c(7,3,4,5)]
    DS_data$V3[which(DS_data$V3=="--")]=0
    DS_data$V4[which(DS_data$V4=="--")]=0
    DS_data$V5[which(DS_data$V5=="--")]=0
    DS_data=xts::xts(DS_data[,-1],order.by = as.Date(DS_data[,1]))
    colnames(DS_data)=c("Sgu","ZZ","Pxi")
    newdata=paste(substr(as.character(stock_names[i]),start = 1,stop = 6),"_DS",".csv",sep = "")
    #newdata=paste(stock_names[i],"_DS",".csv",sep = "")
    write.csv(as.data.frame(DS_data),file = newdata)
  }
}
Divid_Split(".NewHS300.csv")