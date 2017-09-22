import numpy as np
import pandas as pd

games=pd.read_csv("CFB2016_scores.csv",header=None,names=["a_ind","a_score","b_ind","b_score"])
teams=pd.read_csv("TeamNames.txt",header=None,names=["team"])
index=pd.Series(range(1,len(teams)+1))
teams['ind']=index

M=np.zeros((len(teams),len(teams)),dtype=float)
for i in range(len(games)):
    if games.a_score[i]>games.b_score[i]:
        a_wins=1
        b_wins=0
    else:
        b_wins=1
        a_wins=0
    a_prop=(games.a_score[i]+0.0)/(games.a_score[i]+games.b_score[i]+0.0)
    b_prop=(games.b_score[i]+0.0)/(games.a_score[i]+games.b_score[i]+0.0)

    M[games.a_ind[i]-1][games.a_ind[i]-1]=M[games.a_ind[i]-1][games.a_ind[i]-1]+a_wins+a_prop
    M[games.b_ind[i]-1][games.b_ind[i]-1]=M[games.b_ind[i]-1][games.b_ind[i]-1]+b_wins+b_prop
    
    M[games.a_ind[i]-1][games.b_ind[i]-1]=M[games.a_ind[i]-1][games.b_ind[i]-1]+b_wins+b_prop
    M[games.b_ind[i]-1][games.a_ind[i]-1]=M[games.b_ind[i]-1][games.a_ind[i]-1]+a_wins+a_prop
    
sums=M.sum(axis=1)
for i in range(len(M)):
    M[i]=M[i]/sums[i]

iters=10000
w=np.ones((1,len(teams)),dtype=float)/len(teams)
for i in range(iters):
    w=np.dot(w,M)
    top25=np.transpose(np.argsort(w)[0][len(w[0])-25:len(w[0])])
    top25_ord=top25.copy()
    top25_ind=top25.copy()

    for i in range(len(top25)):
        top25_ord[i]=top25[len(top25)-i-1]+1
        top25_ind[i]=top25[len(top25)-i-1]
    top25_ser=pd.DataFrame(top25_ord,columns=['ind'])    
    top25_ser['value']=pd.Series(w[0][top25_ind])
#top25_ser.head()
top25_ser.merge(teams, how='left', left_on='ind', right_on='ind')

w_oth,v=np.linalg.eig(np.transpose(M))
w_oth

import matplotlib.pyplot as plt
plt.plot(dist)
plt.ylabel('Distance between wt and w_inf')
plt.xlabel('Iteration number')
with open(...) as f:
    for line in f:
plt.show()

i=0
X=np.zeros((3012,8447),dtype=float)
with open("nyt_data.txt") as f:
    for line in f:
        curr=line.strip()
        pairs=curr.split(',')
        for pair in pairs:
            mapping=pair.split(':')
            X[int(mapping[0])-1][i]=int(mapping[1])
        i=i+1
print X

from sklearn.preprocessing import normalize
obj = np.zeros(100)
Losses=[]
W=np.random.uniform(low=1.0, high=2.0, size=(3012,25))
H=np.random.uniform(low=1.0, high=2.0, size=(25,8447))

for t in range(100):
    print t
    prod = np.dot(W,H)
    prod[prod==0] = 1e-16
    mat1 = X/prod
    mat2 = W.T.copy()
    for i in range(len(mat2)):
        mat2[i] = normalize(mat2[i][:,np.newaxis],axis=0,norm='l1').ravel()
    H = H *np.dot(mat2,mat1)
    
    prod = np.dot(W,H)
    prod[prod==0] = 1e-16
    mat1 = X/prod
    
    mat3 = H.T.copy()
    for j in range(mat3.shape[1]):
        mat3[:,j] = normalize(mat3[:,j][:,np.newaxis],axis=0,norm='l1').ravel()
    W = W * np.dot(mat1,mat3)
    prod = np.dot(W,H)
    prod[prod==0] = 1e-16
    loss=np.sum(X*-np.log(prod)+prod)
    Losses.append(loss)
    print loss

    import matplotlib.pyplot as plt

plt.plot(range(100),Losses)
plt.xlabel("Iteration number")
plt.ylabel("Objective")
plt.show()

nyt = np.loadtxt("nyt_vocab.dat",dtype=str)
nyt[(Wdf.sort_values(1,ascending=False))['index'][:10]]
wtlist = {}
words = {}
for i in range(25):
    words[i] = nyt[(Wdf.sort_values(i,ascending=False))['index'][:10]]
    wtlist[i] = (Wdf.sort_values(i,ascending=False))[i][:10]