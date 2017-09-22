m=500
N1=np.random.multivariate_normal([0,0],[[1,0],[0,1]],size=m)
N2=np.random.multivariate_normal([3,0],[[1,0],[0,1]],size=m)
N3=np.random.multivariate_normal([0,3],[[1,0],[0,1]],size=m)
m=500
N1=np.random.multivariate_normal([0,0],[[1,0],[0,1]],size=m)
N2=np.random.multivariate_normal([3,0],[[1,0],[0,1]],size=m)
N3=np.random.multivariate_normal([0,3],[[1,0],[0,1]],size=m)

data=[]
for i in range(m):
    k=np.random.choice([0,1,2], size=1, replace=True, p=[0.2,0.5,0.3])
    selected=[N1[i],N2[i],N3[i]][k]
    data.append(selected)
​
def dist(point1,point2):
    return np.power(point1-point2,2).sum()
dist(np.asarray([3,2]),np.asarray([1,0]))

k=4 #clusters
c=[]
for i in range(k):
    c.append(np.random.multivariate_normal([i,i],[[1,0],[0,1]],size=1))
ck=np.zeros(len(data),dtype=int)
iters=20
Losses=[]
for j in range(iters):
    for i in range(len(ck)):
        #print len(ck)
        distances=[]
        for l in range(len(c)):
            distances.append(dist(c[l],data[i]))
        #print distances
        ck[i]=np.argmin(distances)
        
    #print ck    
    #Means updating
    c_new=np.zeros((k,2),dtype=float)
    leng=np.zeros(k,dtype=float)
    for g in range(len(ck)):
        c_new[int(ck[g])]=c_new[int(ck[g])]+data[g]
        #print ck[i]
        leng[ck[g]]=leng[ck[g]]+1
​
    for g in range(len(c)):
        c[g]=np.divide(c_new[g],leng[g])
​
    #Loss Calculation
    Loss=0
    for i in range(len(ck)):
        Loss=Loss+dist(data[i],c[int(ck[i])])
    Losses.append(Loss)
        
Losses3=Losses
Losses

import pandas as pd
pd.DataFrame(data).to_csv('data4.csv')
pd.DataFrame(ck).to_csv('ck4.csv')


mapping
import pandas as pd
import numpy as np
mat=pd.read_csv("ratings.csv",header=None,names=['user_id','mov_id','rating'])
mat_test=pd.read_csv("ratings_test.csv",header=None,names=['user_id','mov_id','rating'])
mapping=pd.read_csv("movies.txt",header=None,names=['movie'])
ids=pd.Series(np.zeros(len(mapping)))
mapping['ids']=ids
for i in range(len(mapping)):
    mapping.set_value(i,'ids',i+1)
mapping.head()

import time
start_time = time.time()
N1=len(np.unique(mat.user_id))
d=10
lamb=1
sig_sq=0.25
users=np.unique(mat.user_id)
N2=len(np.unique(mat.mov_id))
movies=np.unique(mat.mov_id)
​
M=np.zeros((N1,N2),dtype=float)
​
for i in range(len(mat)):
    M[np.where(users==mat.user_id[i])[0][0]][np.where(movies==mat.mov_id[i])[0][0]]=mat.rating[i]
print("--- %s seconds ---" % (time.time() - start_time))


#start_time = time.time()
def run():
    iters=100
​
    means=np.zeros(N1)
    cov=np.identity(N1)
    U=np.random.multivariate_normal(means,cov,d)
​
    means=np.zeros(N2)
    cov=np.identity(N2)
    V=np.random.multivariate_normal(means,cov,d)
​
    add_mat=lamb*sig_sq*np.identity(d)
    Losses=[]
​
    present_whole=np.where(M!=0)
    M_sub_whole=M[present_whole]
​
    #print("--- %s seconds ---" % (time.time() - start_time))
​
    for i in range(iters):
        for j in range(N1):
            present=np.where(M[j]!=0)[0]
​
            V_sub=V[:,present]
            vvt=np.dot(V_sub,np.transpose(V_sub))
            mat1=np.linalg.inv(add_mat+vvt)
            M_sub=np.asmatrix(M[j][present])
            mat2=np.sum(np.multiply(M_sub,V_sub),axis=1)
            U[:,j]=np.transpose(np.dot(mat1,mat2))
        #print("--- %s seconds ---" % (time.time() - start_time))
​
        for j in range(N2):
            present=np.where(M[:,j]!=0)[0]
            U_sub=U[:,present]
            uut=np.dot(U_sub,np.transpose(U_sub))
            mat1=np.linalg.inv(add_mat+uut)
            M_sub=np.asmatrix(M[present,j])
            mat2=np.sum(np.multiply(M_sub,U_sub),axis=1)
            V[:,j]=np.transpose(np.dot(mat1,mat2))
        #print("--- %s seconds ---" % (time.time() - start_time))
​
        UTV=M_sub_whole.copy()
        for j in range(len(present_whole[0])):
            ut=np.asmatrix(U[:,present_whole[0][j]])
            v=np.transpose(np.asmatrix(V[:,present_whole[1][j]]))
            UTV[j]=np.dot(ut,v)[0,0]
        L=-np.square(M_sub_whole-UTV).sum()/(2*sig_sq)
        L=L-lamb*(np.square(U).sum()+np.square(V).sum())/2
        Losses.append(L)
    return Losses,U,V
        #print L
        #print("--- %s seconds ---" % (time.time() - start_time))


L_all=[]
U_all=[]
V_all=[]
​
for i in range(10):
    Losses,U,V=run()
    L_all.append(Losses)
    U_all.append(U)
    V_all.append(V)
    print i
    

M_test=np.asmatrix(mat_test.rating)
pred_M=M_test.copy()
RMSE=[]
for i in range(10):
    U=U_all[i]
    V=V_all[i]
    
    for j in range(len(mat_test)):
        if len(np.where(movies==mat_test.mov_id[j])[0])==0:
            pred_M[0,j]=0.46961
        else:
            ut=np.asmatrix(U[:,np.where(users==mat_test.user_id[j])[0][0]])
            v=np.transpose(np.asmatrix(np.asmatrix(V[:,np.where(movies==mat_test.mov_id[j])[0][0]])))
            pred_M[0,j]=np.dot(ut,v)[0,0]
    rmse=np.sqrt(np.square(M_test-pred_M).sum()/len(mat_test))
    RMSE.append(rmse)
​

three = [50, 485, 182]
U=U_all[3]
V=V_all[3]
​
for k in three:
    present=np.where(movies==k)[0][0]
    v=V[:,present]
    dist=[]
    for i in range(N2):
        if i==present:
            dist.append(10000)
        else:
            v2=V[:,i]
            dist.append(np.linalg.norm(v2-v))


distances=pd.DataFrame(dist,columns=['dist'])
distances['ids']=pd.Series(movies)
distances_sort=distances.sort(columns='dist', axis=0, ascending=True).reset_index(drop=True)
top_dist=distances_sort[0:10]
#print top_dist
top_dist.merge(mapping, how='left', on='ids')