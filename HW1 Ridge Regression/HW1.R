library(readr)
library(reshape2)
X_test <- read_csv("G:/Acads/ML for data science/hw1-data/X_test.csv", 
                   col_names = FALSE)
X_train <- read_csv("G:/Acads/ML for data science/hw1-data/X_train.csv", 
                   col_names = FALSE)
y_test <- read_csv("G:/Acads/ML for data science/hw1-data/y_test.csv", 
                   col_names = FALSE)
y_train <- read_csv("G:/Acads/ML for data science/hw1-data/y_train.csv", 
                   col_names = FALSE)

X_test_mat=as.matrix(X_test)
X_train_mat=as.matrix(X_train)
y_test_mat=as.matrix(y_test)
y_train_mat=as.matrix(y_train)

lambda=c(0:5000)
df_lambda=array(dim=length(lambda))
RMSE=array(dim=length(lambda))
W_mat=matrix(nrow=ncol(X_train),ncol=length(lambda))

#Calculating df_lambda, W_mat, RMSE on Y_test and y_pred on test data
for (i in 1:length(lambda)){
  W_mat[,i]=solve(lambda[i]*diag(ncol(X_train_mat))+((t(X_train_mat))%*%X_train_mat))%*%t(X_train_mat)%*%y_train_mat
  df_lambda[i]=sum(diag(X_train_mat%*%solve(lambda[i]*diag(ncol(X_train_mat))+t(X_train_mat)%*%X_train_mat)%*%t(X_train_mat)))
  y_pred=X_test_mat%*%W_mat[,i]
  RMSE[i]=sqrt(sum((y_pred-y_test_mat)^2)/length(y_pred))
}

W_mat_melt <- melt(W_mat, id=c())
W_mat_melt$df_lambda=0
for (i in 1:length(W_mat_melt$df_lambda)){
  W_mat_melt$df_lambda[i]=df_lambda[W_mat_melt$Var2[i]]
}
df_lambda[W_mat_melt$Var2[10]]
W_mat_melt$Var2[10]

ggplot(data=W_mat_melt)+geom_line(mapping=aes(y=value,x=df_lambda,colour=factor(Var1)))+
  labs(title="Values of W w.r.t df_lambda",x="df_lambda",y="W values")

rmse_frame=data.frame(RMSE,lambda)
ggplot(data=rmse_frame[1:51,])+geom_line(aes(y=RMSE,x=lambda))+labs(title="RMSE on test data w.r.t lambda")

#p=2
X_train_mat_cr=matrix(nrow=nrow(X_train_mat),ncol=13)
X_train_mat_cr[,1:7]=X_train_mat
X_test_mat_cr=matrix(nrow=nrow(X_test_mat),ncol=13)
X_test_mat_cr[,1:7]=X_test_mat
W_mat_cr=matrix(nrow=ncol(X_train_mat_cr),ncol=length(lambda))
for (i in 1:6){
  X_train_mat_cr[,i+7]=X_train_mat[,i]^2
  X_test_mat_cr[,i+7]=X_test_mat[,i]^2
}
for (i in 1:length(lambda)){
  W_mat_cr[,i]=solve(lambda[i]*diag(ncol(X_train_mat_cr))+((t(X_train_mat_cr))%*%X_train_mat_cr))%*%t(X_train_mat_cr)%*%y_train_mat
  #df_lambda[i]=sum(diag(X_train_mat_cr%*%solve(lambda[i]*diag(ncol(X_train_mat_cr))+t(X_train_mat_cr)%*%X_train_mat_cr)%*%t(X_train_mat_cr)))
  y_pred=X_test_mat_cr%*%W_mat_cr[,i]
  RMSE[i]=sqrt(sum((y_pred-y_test_mat)^2)/length(y_pred))
}

rmse_frame_cr=data.frame(RMSE,lambda)
ggplot(data=rmse_frame_cr[1:501,])+geom_line(aes(y=RMSE,x=lambda))+labs(title="RMSE on test data(p=2) w.r.t lambda")

#p=3
X_train_mat_cr3=matrix(nrow=nrow(X_train_mat),ncol=19)
X_train_mat_cr3[,1:13]=X_train_mat_cr
X_test_mat_cr3=matrix(nrow=nrow(X_test_mat),ncol=19)
X_test_mat_cr3[,1:13]=X_test_mat_cr
W_mat_cr3=matrix(nrow=ncol(X_test_mat_cr3),ncol=length(lambda))
for (i in 1:6){
  X_train_mat_cr3[,i+13]=X_train_mat[,i]^3
  X_test_mat_cr3[,i+13]=X_test_mat[,i]^3
}
for (i in 1:length(lambda)){
  W_mat_cr3[,i]=solve(lambda[i]*diag(ncol(X_train_mat_cr3))+((t(X_train_mat_cr3))%*%X_train_mat_cr3))%*%t(X_train_mat_cr3)%*%y_train_mat
  #df_lambda[i]=sum(diag(X_train_mat_cr%*%solve(lambda[i]*diag(ncol(X_train_mat_cr3))+t(X_train_mat_cr3)%*%X_train_mat_cr3)%*%t(X_train_mat_cr3)))
  y_pred=X_test_mat_cr3%*%W_mat_cr3[,i]
  RMSE[i]=sqrt(sum((y_pred-y_test_mat)^2)/length(y_pred))
}

rmse_frame_cr3=data.frame(RMSE,lambda)
ggplot(data=rmse_frame_cr3[1:501,])+geom_line(aes(y=RMSE,x=lambda))+labs(title="RMSE on test data(p=3) w.r.t lambda")
#Ideal value of lambda=21

rmse_frame$p=1
rmse_frame_cr$p=2
rmse_frame_cr3$p=3
rmse_all=rbind(rmse_frame[1:501,],rmse_frame_cr[1:501,],rmse_frame_cr3[1:501,])
ggplot(data=rmse_all)+geom_line(aes(y=RMSE,x=lambda,colour=factor(p)))+labs(title="RMSE w.r.t lambda")
