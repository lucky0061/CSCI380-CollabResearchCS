#RNN and 44features
TradeSignal=function(sec_names){
  idNumber=read.csv(sec_names,header=F)$V1
  #idNumber=substr(as.character(idNumber),start = 1,stop = 6)#".NewHS300.csv"
  Datafiles=paste(idNumber,"_DatasetNew",".csv",sep = "")
  Signal=paste(idNumber,"_TradeSignal_RNN",".csv",sep = "")
  N=length(Datafiles)
  for(j in 1:N){
    model_data=read.csv(Datafiles[j],header = T)
    #model_data=model_data[,-29]#features=43
    predict_signal=NULL
    for(i in 1:350){
      train_set=model_data[(1+5*(i-1)):(250+5*(i-1)),]
      x=train_set[,2:45]
      x=t(x)
      test_set=model_data[(251+5*(i-1)):((255+5*(i-1))),2:45]
      y=train_set[,46,drop=F]
      y=t(y)
      z=t(test_set)
      X=array(unlist(as.list(x[,1:250]))
              ,dim = c(dim(x[1,,drop=FALSE]),44))
      Y=array(y[1,1:250],dim = c(dim(y),1))
      X_test=array(unlist(as.list(z[,1:5]))
                   ,dim = c(dim(z[1,,drop=FALSE]),44))
      model=rnn::trainr(Y=Y,X=X,learningrate=0.01,hidden_dim=c(10,5),network_type ="rnn",sigmoid = "logistic" )
      Y_test=rnn::predictr(model,X_test )
      class_predict=as.vector(ifelse(Y_test>=0.5,1,0))[1:5]
      predict_signal=c(predict_signal,class_predict)
    }
    pre_signal=c(model_data$Label[1:250],predict_signal)
    #mean(pre_signal==model_data$Label)
    pre_signal=xts::xts(pre_signal,order.by=as.Date(model_data$X))
    names(pre_signal)="pre_signal"
    write.csv(as.data.frame(pre_signal),file = Signal[j])
  }
}
TradeSignal(".NewSP500.csv")#".NewHS300.csv"
