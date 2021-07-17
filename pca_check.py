#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 14:04:10 2021

@author: anujkumar
"""

import os
import pandas as pd
import numpy as np
from numpy import cov, diag
import numpy.linalg as la
from numpy.linalg import eig, pinv
import matplotlib.pyplot as plt

from sklearn import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#fdata3 = np.loadtxt("plane-xy-0001.1.680E-04.dat", skiprows=22)
#fdata2 = np.loadtxt("plane-xy-0001.4.800E-05.dat", skiprows=22)
os.chdir("/Users/anujkumar/Documents/RESEARCH/PreProcessingCode/2D_PCA")

tec_file= "plane-xy-0001.2.400E-04.dat"



fdata1 = np.loadtxt(tec_file, skiprows=27)

header= pd.read_csv(tec_file, nrows=21)


#select species 

Y= pd.DataFrame(fdata1)
            
Coord= np.array(Y[[0,1]])
                       
# Selecting species            
Y     =    Y[[6,12,13,16,17]]
Y_min= Y.min()
Y_max= Y.max()

# Selecting domain
smdata= filter_data(fdata1, -4.0, 4.0, 0.0, 10.0)

plot_contour(smdata, 6)
# Randomly selecting points

rand_data= rand_select(smdata, 5000)

rcoord= rand_data[:,0:2]

unst_plot(rcoord, rand_data[:,6])


write_sol(rand_data, header) 



f= open("yminmax.dat", "w")
np.savetxt(f,np.array(Y_max))
np.savetxt(f,np.array(Y_min))
f.close()







### THIS IS STEP IS TO DONE AFTER PCA- ANN TABULATION


Q_m,eig_val,cum_var,PCsAll= dopca(fdata1)
    
rot_mat_file= "pc_clookup.dat"

rot_mat= np.loadtxt(rot_mat_file, skiprows=13)
rot_mat= rot_mat.T
    
PCsrot= PCsAll.dot(rot_mat)

for i in range(5):
    unst_plot(Coord, PCsAll[i])
    plt.show()
    
for i in range(5):
    unst_plot(Coord, PCsrot[i])
    plt.show()    


write_data(fdata1)

#dlength= fdata1.shape[0]
#dwidth= fdata1.shape[1]
#
#Y1= pd.DataFrame(fdata1)
#
#Coord= Y1[[0,1]]

#
##Y= pd.DataFrame(np.vstack((fdata1, fdata2, fdata3)))
#
#Y     =    Y1[[6,12,13,16,17]]
#
#
#min_max_scaler = pp.MinMaxScaler()
#Y_n = pd.DataFrame(2.0*min_max_scaler.fit_transform(Y)-1.0)
#
#
#
#
#C = cov(Y_n.T)
#eig_val, Q_matrix = eig(C)
#eig_values= -np.sort(-eig_val)
#ind= np.argsort(-eig_val)
#Q_matrix= Q_matrix[:,ind]
#
#PCsAll   = pd.DataFrame(Y_n.dot(Q_matrix))
#
#N_pc= 4
#pca = PCA(n_components=N_pc)
#pca.fit(Y_n)
#cumul_var =np.cumsum(np.round(pca.explained_variance_ratio_, decimals=5)*100)
#print('Cumulative Variance with' + str(N_pc) + ' PCs is: ', cumul_var[N_pc-1])
#### cumul_var =np.cumsum(np.round(eig_values, decimals=5)*100)
#### for this, we are using the pca function with sklearn
#plt.clf()
#plt.ylabel('% Variance',fontsize=12)
#plt.xlabel('# of principal components',fontsize=12)
#plt.title('Scree Plot',fontsize=14)
#plt.xlim(0,7)
#plt.ylim(80,100.5)
#plt.style.context('seaborn-whitegrid')
#plt.plot(cumul_var,'k-o',markerfacecolor='None',markeredgecolor='k')


#pccut1= PCsAll.iloc[0:dlength,:]
#pccut2= PCsAll.iloc[dlength:2*dlength, :]
#pccut3= PCsAll.iloc[2*dlength:, :]
#
#file = open("pcplots.dat","w")
#
#all_data = pd.DataFrame(np.hstack([Coord, pccut1, pccut2, pccut3]))
#
#np.savetxt(file,all_data)
#file.close()






def dopca(fdata1):
    dlength= fdata1.shape[0]
    dwidth= fdata1.shape[1]
            
    Y= pd.DataFrame(fdata1)
            
    Coord= Y[[0,1]]
            
           
            
    Y     =    Y[[6,12,13,16,17]]
            
            
    min_max_scaler = pp.MinMaxScaler()
    Y_n = pd.DataFrame(2.0*min_max_scaler.fit_transform(Y)-1.0)
            
            
            
    C = cov(Y_n.T)
    eig_val, Q_matrix = eig(C)
    eig_values= -np.sort(-eig_val)
    ind= np.argsort(-eig_val)
    Q_matrix= Q_matrix[:,ind]
    
    print(eig_values)
    PCsAll   = pd.DataFrame(Y_n.dot(Q_matrix))
    
    
    N_pc= 5
    pca = PCA(n_components=N_pc)
    pca.fit(Y_n)
    cumul_var =np.cumsum(np.round(pca.explained_variance_ratio_, decimals=5)*100)
    print('Cumulative Variance with' + str(N_pc) + ' PCs is: ', cumul_var[N_pc-1])
    ### cumul_var =np.cumsum(np.round(eig_values, decimals=5)*100)
    ### for this, we are using the pca function with sklearn
    plt.clf()
    plt.ylabel('% Variance',fontsize=12)
    plt.xlabel('# of principal components',fontsize=12)
    plt.title('Scree Plot',fontsize=14)
    plt.xlim(0,7)
    plt.ylim(85,100.5)
    plt.style.context('seaborn-whitegrid')
    plt.plot(cumul_var,'k-o',markerfacecolor='None',markeredgecolor='k')

        
    
    return Q_matrix, eig_values, cumul_var, PCsAll




def write_data(fdata1):
    
    Q_m,eig_val,cum_var,PCsAll= dopca(fdata1)
    
    rot_mat_file= "pc_clookup.dat"

    rot_mat= np.loadtxt(rot_mat_file, skiprows=13)
    rot_mat= rot_mat.T
    
    PCsrot= PCsAll.dot(rot_mat)
    
    PCsrot.to_csv('init_PCs.dat', index= False, sep= " ")    
    
    vel= pd.DataFrame(fdata1[:,2:5])
    press= pd.DataFrame(fdata1[:,7])
    
    vel.to_csv('init_vel.dat', index= False, sep= " ")  
    press.to_csv('init_press.dat', index= False, sep= " ")  



    return





def corrcomp(Qm1, Qm2):
    
    lsize= Qm1.shape[0]
    intm= (Qm1.T).dot(Qm2)
    
    corrcf= np.diag(intm)
    
    return np.abs(corrcf)






#Qm1, eig1= dopca(fdata1)
#Qm2, eig2= dopca(fdata2)
#Qm3, eig3= dopca(fdata3)
#
#corr12= corrcomp(Qm1, Qm2)
#corr23= corrcomp(Qm2,Qm3)
#corr13= corrcomp(Qm1,Qm3)
#corr1= corrcomp(Qm1,Q_matrix)
#corr2= corrcomp(Qm2, Q_matrix)
#corr3= corrcomp(Qm3, Q_matrix)
#

#xaxis= np.array([1,2,3,4,5])
#
#plt.semilogy(xaxis,eig3,'bo', label='data 1')
#plt.semilogy(xaxis, Seig3, 'go', label= 'data 2')
#plt.semilogy(xaxis, vseig, 'ro', label= 'data 3')
##plt.semilogy(xaxis, eig_values, 'ko', label= 'data all')
#plt.legend(loc="upper right")
#
#plt.show




#plt.plot(xaxis, corr1,'bo-', label='data 1')
#plt.plot(xaxis, corr2, 'go-', label= 'data 2')
#plt.plot(xaxis, corr3, 'ro-', label= 'data 3')
#
#plt.legend(loc="upper right")
#
#plt.show





def filter_data(fdata, ymin, ymax, xmin, xmax):
    
    dlength= fdata.shape[0]
    dwidth= fdata.shape[1]
    
    
    #ymin= -4.0
    #ymax= 4.0
    
    #xmin= 0.0
    #xmax= 8.5
    
    fshape= np.array([0,0])
    
    sorted_list= list()
    
    for i in range(dlength):
        xdata=fdata[i][0]
        ydata= fdata[i][1]
        if ((ydata>=ymin)and(ydata<=ymax)):
            if ((xdata>=xmin)and(xdata<=xmax)):
                sorted_list.append(fdata[i])
                
   
    
    
    
    small_fdata= np.array(sorted_list)
    fshape[0]= small_fdata.shape[0]
    fshape[1]= small_fdata.shape[1]
    
    
    return small_fdata




def plot_contour(smdata, val):
    
    dlength= smdata.shape[0]
    dwidth= smdata.shape[1]
    first_val= smdata[0][0]
    
    for i in range(1, dlength):
        if (first_val==smdata[i][0]):
            plot_len= i
            break
    #print(dlength, plot_len)    
        
    plot_wid= int(dlength/plot_len)
    

    xx= np.zeros((plot_len, plot_wid))
    yy= np.zeros((plot_len, plot_wid))
    plot_val= np.zeros((plot_len, plot_wid))
    
    
    for j in range(plot_wid):
        for i in range(plot_len):
            m= i+j*plot_len
            xx[i][j]= smdata[m][0]
            yy[i][j]= smdata[m][1]
            plot_val[i][j]= smdata[m][val]
            
    
    print(plot_val.shape) 
    #print(xx) 
    #print(yy)      
    plt.contourf(xx,yy, plot_val, 20, cmap=plt.cm.jet, extend='both')
    #plt.axis('square');
    #plt.cmap.set_over('red')
    #plt.contourf(plot_val)
    plt.colorbar()
    plt.show()            
            
            
        
    
    return







def write_sol(fdata, header):
    
    species= header.T[[0,1,5,6,7,12,13,14,15,16,17,18,19,20]]
    #species.reset_index(drop=True,inplace=True)
    
    
    Y= pd.DataFrame(fdata)            
            
    Y     =    Y[[0,1,5,6,7,12,13,14,15,16,17,18,19,20]]
    
    Y= species.append(Y)

    
    #file= open("Solution.dat", "w")
    #np.savetxt(file, Y)
    #file.close()
    Y.to_csv('Solution.dat', index= False, sep= " ")    
     
#file = open("pcplots.dat","w")
#
#all_data = pd.DataFrame(np.hstack([Coord, pccut1, pccut2, pccut3]))
#
#np.savetxt(file,all_data)
#file.close()




def rand_select(fdata, npoints):
    
    indices= np.random.choice(fdata.shape[0], npoints, replace=False)
    #print(np.sort(indices))
    rand_data= fdata[np.sort(indices)]
    return rand_data







def unst_plot(xy, val):
    xx= xy[:,0]
    yy= xy[:,1]
    
    pval= np.array(val)
    
    plt.tricontourf(xx,yy,pval,20, cmap=plt.cm.jet, extend='both')
    plt.colorbar()
    plt.show
    
    return




plot_contour(fdata1)

smdata1= filter_data(fdata1, -2, 2, 0, 8)
smdata2= filter_data(fdata1, -2, 2, 0, 1)
smdata3= filter_data(fdata1, 0, 2, 0, 1)
smdata4= filter_data(fdata1, -1, 1, 4.5, 5.5)
smdata5= filter_data(fdata1, -1, 0.3, 3, 5)

plot_contour(smdata1)
plot_contour(smdata2)
plot_contour(smdata3)
plot_contour(smdata4)
plot_contour(smdata5)


qmg, eigg, cum_varg= dopca(fdata1)
qm1, eig1, cum_var1= dopca(smdata1)
qm2, eig2, cum_var2= dopca(smdata2)
qm3, eig3, cum_var3= dopca(smdata3)
qm4, eig4, cum_var4= dopca(smdata4)
qm5, eig5, cum_var5= dopca(smdata5)


xaxis= np.array([1,2,3,4,5])

plt.semilogy(xaxis,eigg,'bo', label='global data')
plt.semilogy(xaxis, eig1, 'go', label= 'smaller_version')
plt.semilogy(xaxis, eig2, 'yo', label= 'Initial')
plt.semilogy(xaxis, eig3, 'ko', label= 'Initial_half')
plt.semilogy(xaxis, eig4, 'mo', label= 'Tip')
plt.semilogy(xaxis, eig5, 'ro', label= 'Wrinkle')
#plt.semilogy(xaxis, eig_values, 'ko', label= 'data all')
plt.legend(loc="upper right")

plt.show



plt.clf()
plt.ylabel('% Variance',fontsize=12)
plt.xlabel('# of principal components',fontsize=12)
plt.title('Scree Plot',fontsize=14)
plt.xlim(0,5)
plt.ylim(80,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(cum_varg,'b-o', label='global data')
plt.plot(cum_var1,'g-o',label= 'smaller_version')
plt.plot(cum_var2,'y-o', label= 'Initial')
plt.plot(cum_var3,'k-o', label= 'Initial_half')
plt.plot(cum_var4,'m-o', label= 'Tip')
plt.plot(cum_var5, 'r-o', label= 'Wrinkle')
plt.legend(loc="upper right")
plt.show()












corr12= corrcomp(qm1, qm2)
corr23= corrcomp(qm2,qm3)
corr13= corrcomp(qm1,qm3)

plt.plot(xaxis, corr12,'bo-', label='data 12')
plt.plot(xaxis, corr23, 'go-', label= 'data 23')
plt.plot(xaxis, corr13, 'ro-', label= 'data 13')

plt.legend(loc="upper right")

plt.show


