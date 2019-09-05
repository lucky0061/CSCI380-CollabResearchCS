#Random Forest
TradeSignal=function(sec_names){
  idNumber=read.csv(sec_names,header=F)$V1
  #idNumber=substr(as.character(idNumber),start = 1,stop = 6)#".NewHS300.csv"
  Datafiles=paste(idNumber,"_DatasetNew",".csv",sep = "")
  Signal=paste(idNumber,"_TradeSignal_RF",".csv",sep = "")
  N=length(Datafiles)
  for(j in 1:N){
    model_data=read.csv(Datafiles[j],header = T)
    l=dim(model_data)[2]
    model_data=data.frame(model_data[,-l],as.factor(model_data$Label))
    colnames(model_data)[l]="Label"
    #model_data=model_data[,-29]
    predict_signal=NULL
    for(i in 1:350){
      train_set=model_data[(1+5*(i-1)):(250+5*(i-1)),2:46]
      test_set=model_data[(251+5*(i-1)):((255+5*(i-1))),2:45]
      model=randomForest::randomForest(Label~.,data=train_set)
      class_predict=unname(predict(model,test_set))
      predict_signal=c(predict_signal,class_predict)
    }
    pre_signal=c(model_data$Label[1:250],predict_signal)
    mean(pre_signal==as.numeric(model_data$Label))
    pre_signal=xts::xts(pre_signal,order.by=as.Date(model_data$X))-1
    names(pre_signal)="pre_signal"
    write.csv(as.data.frame(pre_signal),file = Signal[j])
  }
}
TradeSignal(".NewSP500.csv")#".NewHS300.csv"
