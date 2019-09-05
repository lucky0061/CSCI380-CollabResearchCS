# TECH INDEX/ 44 tech indicators
label_func=function(stock,k,add_label=NULL){
  cl=stock[,"Close"]
  re=quantmod::dailyReturn(cl,type = "log")
  n=length(re)
  cor_re=zoo::coredata(re)
  for(i in 1:(n-k)){
    label=ifelse(mean(re[(i+1):(i+k)])>=0,1,0)
    add_label=c(add_label,label)
  }
  return(xts::xts(add_label,order.by = zoo::index(re[-((n-k+1):n)])))
}
#--------------chaikin oscillator---------------------------------
CO=function(H,L,C,V){
  N=length(H)
  date=zoo::index(H)
  H=as.vector(H)
  L=as.vector(L)
  C=as.vector(C)
  V=as.vector(V)
  AD=NULL
  AD[1]=((C[1]-L[1])-(H[1]-C[1]))*V[1]/(H[1]-L[1]+0.01)
  for(i in 2:N ){
    AD[i]=AD[i-1]+(((C[i]-L[i])-(H[i]-C[i]))/(H[i]-L[i]+0.01))*V[i]
  }
  CO1=TTR::EMA(AD,3)-TTR::EMA(AD,10)
  CO2=xts::xts(CO1,order.by =date)
  return(CO2)
}
#------------------disparity--------------------------------
DIS=function(C){
  return((C/TTR::SMA(C,20))*100)
}
#----------------ease of movement-----------------
EOM=function(H,L,V){
  N=length(H)
  date=zoo::index(H)
  H=as.vector(H)
  L=as.vector(L)
  V=as.vector(V)
  emo=NULL
  emo[1]=(H[1]+L[1])/2
  for(i in 2:N){
    emo[i]=((H[i]+L[i])/2-(H[i-1]+L[i-1])/2)*(H[i]-L[i])/V[i]
  }
  emo=xts::xts(emo,order.by = date)
  return(emo)
}
#----------------------force index---------------
FI=function(C,V){
  fi=NULL
  N=length(C)
  date=zoo::index(C)
  C=as.vector(C)
  V=as.vector(V)
  fi[1]=C[1]
  for(i in 2:N){
    fi[i]=(C[i]-C[i-1])*V[i]
  }
  fi=TTR::SMA(fi,2)
  fi=xts::xts(fi,order.by = date)
  return(fi)
}
#------------------MA oscillator----------------------------
MAO=function(C){
  mao=TTR::SMA(C,12)-TTR::SMA(C,26)
  return(mao)
}
#------------------mass index momentum----------------------
MI=function(H,L){
  r=H-L
  ema1=TTR::EMA(r,9)
  ema2=(TTR::EMA(r,9))^2
  x=ema1/ema2
  mi=TTR::runSum(x,9)
  return(mi)
}
#-----------------momentum--------------------------------
MOM=function(C){
  N=length(C)
  date=zoo::index(C)
  C=as.vector(C)
  mom=NULL
  for(i in 10:N){
    mom[i]=(C[i]/C[i-9])*100
  }
  mom=xts::xts(mom,date)
  return(mom)
}
#------------------Net change oscillator------------------
NCO=function(C){
  nco=TTR::momentum(C,12)
  return(nco)
}
#-----------------------price Oscillator------------------
PO=function(C){
  po=(TTR::SMA(C,5)-TTR::SMA(C,10))/(TTR::SMA(C,5))
  return(po)
}
#-------------------------psychology------------------
PSY=function(C){
  N=length(C)
  date=zoo::index(C)
  C=as.vector(C)
  psy=NULL
  psy[10]=sum(na.omit(TTR::ROC(C[1:10],n=1))>=0)/9
  for(i in 11:N){
    psy[i]=sum(na.omit(TTR::ROC(C[(i-10):i],n=1))>=0)/10
  }
  psy=xts::xts(psy,order.by = date)
  return(psy) 
}
#----------------------------relative momentum index----------------
RMI=function(C){
  N=length(C)
  date=zoo::index(C)
  C=as.vector(C)
  mo=TTR::momentum(C,1)
  rmi=NULL
  mo1=na.omit(mo[1:14])
  rmi[14]=sum(mo1[mo1>=0])/(sum(mo1[mo1>=0])+sum(mo1[mo1<0])+0.01)
  for(i in 15:N){
    mo2=mo[(i-13):i]
    rmi[i]=sum(mo2[mo2>=0])/(sum(mo2[mo2>=0])+sum(mo2[mo2<0])+0.01)
  }
  rmi=xts::xts(rmi,order.by = date)
  return(rmi)
}
#---------------------Volume Moveing Average---------------------
VOf=function(V){
  Vm=TTR::EMA(V,12)
  Vn=TTR::EMA(V,26)
  VO=((Vm-Vn)/Vn)*100
  return(VO)
}
#jisuan beta value
Beta=function(rets,ret500){
  rp=rets+ret500
  rm=rets-ret500
  m1=fGarch::garchFit(~1+garch(1,1),data=rp,trace = F)
  m2=fGarch::garchFit(~1+garch(1,1),data=rm,trace = F)
  m3=fGarch::garchFit(~1+garch(1,1),data=ret500,trace = F)
  vp=fBasics::volatility(m1)
  vm=fBasics::volatility(m2)
  vsp=fBasics::volatility(m3)
  beta=(vp^2-vm^2)/(4*vsp^2)
  return(beta)
}
Tech_Indi=function(namesfile,add=NULL,addnames=NULL){
  idNumber=read.csv(namesfile,header=F)$V1
  #idNumber=substr(as.character(idNumber),start = 1,stop = 6)#".NewHS300.csv"
  infiles=paste(idNumber,"_New",".csv",sep = "")
  outfiles=paste(idNumber,"_TechIndi",".csv",sep = "")
  for(j in 1:length(infiles)){
    Newstock=read.csv(infiles[j],header = T)
    Newstock_Pri=Newstock[,2:6]
    Newstock_Date=Newstock[,1]
    Newstock=xts::xts(Newstock_Pri,order.by=as.Date(Newstock_Date))
    #colnames(Newstock)=c("Close","High","Low","Open","Vol")
    Newstock=Newstock[Newstock$Low>0,]
    #hs300index=read.csv(".hs300.csv",header = T)
    #hs300index=xts::xts(hs300index[,2:6],order.by = as.Date(hs300index[,1]))
    #hs300index=hs300index[zoo::index(Newstock)]
    CL=Newstock[,"Close"]
    ret_sto=TTR::ROC(CL)[-1]
    #HS_Close=hs300index[,1]
    #ret_hs300=TTR::ROC(HS_Close)[-1]
    VO=Newstock[,"Volume"]
    HLC=Newstock[,c("High","Low","Close")]
    ATR=TTR::ATR(HLC,n=14,maType="SMA")$atr
    ADX=TTR::ADX(HLC,14)[,4]#average directional index
    OBV=TTR::OBV(CL,VO)#on balance volume
    WR=TTR::WPR(HLC,14)#William%R
    RSI=TTR::RSI(CL,14)#Relative Strength Index
    CMF=TTR::CMF(HLC,VO,n=20)#Chaikin Money Flow
    BBDN=TTR::BBands(HLC)[,1]
    BBMA=TTR::BBands(HLC)[,2]
    BBUP=TTR::BBands(HLC)[,3]
    BandPer=(CL-BBDN)/(BBUP-BBDN)#Band%b
    BandWid=(BBUP-BBDN)/BBMA#Band width
    CO=CO(HLC[,1],HLC[,2],HLC[,3],VO)#chaikin Volatulity
    DIS=DIS(HLC[,3])#disparity
    #DPO=TTR::DPO(CL,n=14)#Detrended price oscillator
    EOM=EOM(HLC[,1],HLC[,2],VO)#Ease of movement
    FI=FI(HLC[,3],VO)#Force Index
    MAO=MAO(HLC[,3])#MA Oscillator
    MFI=TTR::MFI(HLC,VO,14)#Money flow index
    MI=MI(HLC[,1],HLC[,3])#Mass Index
    MOM=MOM(CL)# momentum
    NCO=NCO(CL)# net change oscillator
    PO=PO(CL)#price oscillator
    PSY=PSY(CL)#psychology
    RMI=RMI(CL)#relative momentum index
    ROC=TTR::ROC(CL,12)# rate of change
    SROC=(TTR::EMA(CL,20)/TTR::EMA(CL,10))*100#smoothed rate of change
    SONAR=TTR::momentum(TTR::EMA(CL,25),25)#snoar
    SONSIG=TTR::EMA(SONAR,9)#snoar signal
    TRIX=TTR::TRIX(CL,12)[,1]#
    VMA=TTR::SMA(VO,20)#Volume Moving Average
    VOS=VOf(VO) #Volume Oscillator
    VROC=TTR::ROC(VO,14)#volume rate of change
    ret=quantmod::dailyReturn(CL)
    Return=TTR::runMean(ret,14)
    Sigma=TTR::runSD(ret,14)
    CCI=TTR::CCI(HLC,14)
    RSV=TTR::stoch(HLC)[,1]
    Kvalue=TTR::EMA(RSV,n=2,ratio = 1/3)
    Dvalue=TTR::EMA(Kvalue,n=2,ratio = 1/3)
    Jvalue=3*Kvalue-2*Dvalue
    MACD=TTR::MACD(CL,nFast = 12,nSlow = 26,nSig = 9)[,1]
    AD=TTR::chaikinAD(HLC,VO)
    VOLA=TTR::chaikinVolatility(Newstock[,c("High","Low")],n=10)
    NBIAS=100*(CL-TTR::SMA(CL,6))/TTR::SMA(CL,6)
    Ret=TTR::ROC(CL)
    SMA_5=TTR::SMA(CL,n=5)
    SMA_10=TTR::SMA(CL,n=10)
    EMA_5=TTR::EMA(CL,5)
    EMA_10=TTR::EMA(CL,10)
    #Beta=Beta(rets = ret_sto,ret500 = ret_hs300)
    #Beta=xts::xts(Beta,order.by = zoo::index(ret_sto))
    Label=label_func(Newstock,1)
    output0=xts::cbind.xts(ATR,ADX,OBV,WR,RSI,CMF,BandPer,BandWid,CO,DIS,EOM,FI,MAO,MFI,MI,MOM,NCO,PO,PSY,RMI,ROC,SROC,SONAR,SONSIG,TRIX,VMA,VOS,VROC,Return,Sigma,CCI,RSV,Kvalue,Dvalue,Jvalue,MACD,AD,VOLA,NBIAS,Ret,SMA_5,SMA_10,EMA_5,EMA_10)
    output=xts::cbind.xts(output0,Label)
    #output1_5=xts::lag.xts(output0,k=1:5)
    #output=xts::cbind.xts(output0,output1_5,Label)
    colname0=c("ATR","ADX","OBV","WR","RSI","CMF","BandPer","BandWid","CO","DIS","EOM","FI","MAO","MFI","MI","MOM","NCO","PO","PSY","RMI","ROC","SROC","SONAR","SONSIG","TRIX","VMA","VOS","VROC","Return","Sigma","CCI","RSV","Kvalue","Dvalue","Jvalue","MACD","AD","VOLA","NBIAS","Ret","SMA_5","SMA_10","EMA_5","EMA_10")
    #colname1=paste(colname0,"_lag_1",sep = "")
    #colname2=paste(colname0,"_lag_2",sep = "")
    #colname3=paste(colname0,"_lag_3",sep = "")
    #colname4=paste(colname0,"_lag_4",sep = "")
    #colname5=paste(colname0,"_lag_5",sep = "")
    colnames(output)=c(colname0,"Label")
    output1=na.omit(output)
    write.csv(as.data.frame(output1),file = outfiles[j])
    #write.csv(as.data.frame(output1),file = "try_AAtechhindi.csv")
    #output1=na.omit(output)
    #n=length(zoo::index(output1))
    #if(n>=1750){
    #output2=output1[(n-1750+1):n,]
    #write.csv(as.data.frame(output2),file = outfiles[j])
    #}
  }
}
Tech_Indi(".NewSP500.csv")#".NewHS300.csv"