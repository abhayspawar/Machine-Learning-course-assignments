X_train_gp=pd.read_csv("G:/Acads/ML for data science/HW3/gaussian_process/X_train.csv",header=None).as_matrix()
X_test_gp=pd.read_csv("G:/Acads/ML for data science/HW3/gaussian_process/X_test.csv",header=None).as_matrix()
y_train_gp=pd.read_csv("G:/Acads/ML for data science/HW3/gaussian_process/y_train.csv",header=None)
y_test_gp=pd.read_csv("G:/Acads/ML for data science/HW3/gaussian_process/y_test.csv",header=None)
y_train_gp=y_train_gp[0].copy().as_matrix()
y_test_gp=y_test_gp[0].copy().as_matrix()
#y_test_gp

#X_train_gp[0]
#Train data kernel
def gaussian_process(X_train_gp,X_test_gp,y_train_gp,b,sigmasq):
    Kn=np.ones((len(X_train_gp),len(X_train_gp)))
    Knx=np.ones((len(X_test_gp),len(X_train_gp)))

    for i in range(len(X_train_gp)):
        for j in range(i+1,len(X_train_gp)):
            Kn[i][j]=np.exp(-1*np.square(X_train_gp[i]-X_train_gp[j]).sum()/b)
            Kn[j][i]=Kn[i][j]
    
    for i in range(len(X_test_gp)):
        for j in range(len(X_train_gp)):
            Knx[i][j]=np.exp(-1*np.square(X_test_gp[i]-X_train_gp[j]).sum()/b)
    y_pred=np.dot(np.dot(Knx,np.linalg.inv(sigmasq*np.identity(len(X_train_gp))+Kn)),y_train_gp)
    return y_pred        

def RMSE(y_pred,y_actual):
    return np.sqrt(np.square(y_pred-y_actual).sum()/len(y_pred))

B=range(5,16,2)
SIGMASQ=np.linspace(0.1,1,10)
rmse=np.zeros((len(B),len(SIGMASQ)))
for i in range(len(B)):
    for j in range(len(SIGMASQ)):
        rmse[i][j]=RMSE(y_test_gp,gaussian_process(X_train_gp,X_test_gp,y_train_gp,B[i],SIGMASQ[j]))
rmse
#RMSE(y_train_gp,y_train_gp)

RMSE(y_test_gp,gaussian_process(X_train_gp,X_test_gp,y_train_gp,11,0.1))
#print X_train_gp

#d part
y_pred_train=gaussian_process(X_train_gp[:,3],X_train_gp[:,3],y_train_gp,5,2)

pd.DataFrame(y_pred_train).to_csv("y_pred_train_ML.csv")


import numpy as np
import pandas as pd
X_train=pd.read_csv("G:/Acads/ML for data science/HW3/boosting/X_train.csv",header=None)
X_test=pd.read_csv("G:/Acads/ML for data science/HW3/boosting/X_test.csv",header=None)
y_train=pd.read_csv("G:/Acads/ML for data science/HW3/boosting/y_train.csv",header=None)
y_test=pd.read_csv("G:/Acads/ML for data science/HW3/boosting/y_test.csv",header=None)
X_train['5']=1
X_test['5']=1

y_train=y_train[0].copy()
y_test=y_test[0].copy()

def error_rate(y_actual,y_predicted):
    error=0.0
    for i in range(len(y_actual)):
        if y_actual[i]!=y_predicted[i]:
            error=error+1
    return (error/len(y_actual))


N=X_train.shape[0]
w=np.ones(N)*(1.0/N)
total=np.zeros(N)
T=1500
ALPHA=[]
coefs=[]
epsilons=[]
for i in range(T):
    sample=np.random.choice(range(N), size=N, replace=True, p=w)
    B=X_train.loc[sample[0]:sample[0]]
    y_b=y_train.copy()
    y_b[0]=y_train[sample[0]]
    total[sample[0]]=total[sample[0]]+1
    for j in range(1,len(sample)):
        frame=[B, X_train.loc[sample[j]:sample[j]]]
        B=pd.concat(frame)
        y_b[j]=y_train[sample[j]]
        total[sample[j]]=total[sample[j]]+1
    B=B.as_matrix()
    #print B.shape
    Bt=np.transpose(B)
    coef=np.ndarray.flatten(np.dot(np.dot(np.linalg.inv(np.dot(Bt,B)),Bt),y_b))
    y_pred=np.sign(np.dot(coef,np.transpose(X_train)))
    
    epsilon=0.0
    for j in range(len(y_b)):
        if y_train[j]!=y_pred[j]:
            epsilon=epsilon+w[j]
    print epsilon
    if epsilon>0.5:
        #epsilon=1-epsilon
        #alpha=-1*alpha
        coef=-1*coef
        epsilon=0.0
        y_pred=-1*y_pred
        for k in range(len(y_b)):
            if y_train[k]!=y_pred[k]:
                epsilon=epsilon+w[k]
        
    
    alpha=(np.log((1-epsilon)/epsilon))/2
    
    epsilons.append(epsilon)
    ALPHA.append(alpha)
    coefs.append(coef)

    for j in range(len(y_b)):
        w[j]=w[j]*np.exp(-1*alpha*y_train[j]*y_pred[j])
    w=w/np.sum(w)
#Changes:
#Save epsilon as well

y_train_pred=np.zeros(len(y_train))
y_test_pred=np.zeros(len(y_test))
accuracy_train=[]
accuracy_test=[]
for i in range(len(ALPHA)):
    y_train_pred=y_train_pred+ALPHA[i]*np.sign(np.dot(coefs[i],np.transpose(X_train)))
    y_train_pred_out=np.sign(y_train_pred)
    accuracy_train.append(error_rate(y_train,y_train_pred_out))
    
    y_test_pred=y_test_pred+ALPHA[i]*np.sign(np.dot(coefs[i],np.transpose(X_test)))
    y_test_pred_out=np.sign(y_test_pred)
    accuracy_test.append(error_rate(y_test,y_test_pred_out))

accuracy_train_df=pd.DataFrame(accuracy_train)
accuracy_test_df=pd.DataFrame(accuracy_test)

frame=[accuracy_train_df,accuracy_test_df]
df=pd.concat(frame,axis=1)
df.head()
df.to_csv("test_train_accuracy.csv")

#Upper bound
UB=[]
for i in range(len(epsilons)):
    ub=0
    for j in range(i+1):
        ub=ub+(0.5-epsilons[j])*(0.5-epsilons[j])
    UB.append(np.exp(-2.0*ub))
    
len(UB)
pd.DataFrame(UB).to_csv("epsilon_ML.csv")

len(total)
pd.DataFrame(total).to_csv("histogram_ML.csv")
pd.DataFrame(epsilons).to_csv("epsilons_actual_ML.csv")
pd.DataFrame(ALPHA).to_csv("aplha_ML.csv")
