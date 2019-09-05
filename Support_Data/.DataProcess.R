#data preprocess and creat a new dataset
f=function(x){
   return((x-min(x,na.rm = T))/(max(x,na.rm = T)-min(x,na.rm = T)))
}
DataProcess=function(sec_names){
   idNumber=read.csv(sec_names,header=F)$V1
   #idNumber=substr(as.character(idNumber),start = 1,stop = 6)#".NewHS300.csv"
   RawFile=paste(idNumber,".csv",sep = "")
   TechIndi=paste(idNumber,"_TechIndi",".csv",sep = "")
   Dataset=paste(idNumber,"_DatasetNew",".csv",sep = "")
   N=length(idNumber)
   for(i in 1:N){
      TechData=read.csv(TechIndi[i],header = T)
      #RawStock=read.csv(RawFile[i],header = T)
      n=ncol(TechData)
      #TechNaomit=na.omit(TechData)
      Features=TechData[,-c(1,n)]
      Normalize=apply(Features,MARGIN = 2,FUN = f)
      #ScaleData=scale(Features)
      Scale_Dataset=xts::xts(cbind(Normalize,Label=TechData[,n]),order.by=as.Date(TechData[,1]))
      #Scale_Dataset=na.omit(Scale_Dataset)
      Data1=Scale_Dataset
      data2=Data1["/2017-12-31"]
      n=dim(data2)[1]
      data3=data2[(n-2000+1):n,]
      write.csv(as.data.frame(data3),file = Dataset[i])
   }
}
DataProcess(".NewSP500.csv")#".NewHS300.csv"
