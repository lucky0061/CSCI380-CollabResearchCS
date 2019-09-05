#XGBoost.R,parameter:booster=gbtree;objective=binary:logistic;eta=0.3;max_depth=10;nround=15;
TradeSignal=function(sec_names){
  idNumber=read.csv(sec_names,header=F)$V1
  #idNumber=substr(as.character(idNumber),start = 1,stop = 6)#".NewHS300.csv"
  Datafiles=paste(idNumber,"_DatasetNew",".csv",sep = "")
  Signal=paste(idNumber,"_TradeSignal_XGB",".csv",sep = "")
  N=length(Datafiles)
  for(j in 1:N){
    model_data=read.csv(Datafiles[j],header = T)
    predict_signal=NULL
    for(i in 1:350){
      train_set=model_data[(1+5*(i-1)):(250+5*(i-1)),]
      test_set=model_data[(251+5*(i-1)):((255+5*(i-1))),2:45]
      test_set=as.matrix(test_set)
      x1=train_set[,2:45]
      x1=as.matrix(x1)
      y1=train_set$Label
      bst=xgboost::xgboost(data = x1,label = y1,nrounds = 15,max_depth=10,eta=0.3,objective="binary:logistic")
      pred=predict(bst,test_set)
      class_predict=as.vector(ifelse(pred>=0.5,1,0))[1:5]
      predict_signal=c(predict_signal,class_predict)
    }
    pre_signal=c(model_data$Label[1:250],predict_signal)
    mean(pre_signal==model_data$Label)
    pre_signal=xts::xts(pre_signal,order.by=as.Date(model_data$X))
    names(pre_signal)="pre_signal"
    write.csv(as.data.frame(pre_signal),file = Signal[j])
  }
}
TradeSignal(".NewSP500.csv")#".NewHS300.csv"