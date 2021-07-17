# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:33:52 2020

@author: Tarek
"""
import os
from sklearn import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from numpy import cov, diag
from numpy.linalg import eig, pinv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Input
import random
from matplotlib.backends.backend_pdf import PdfPages

# Custom activation function
from keras.layers import Activation
#from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

#=====================================================================%
# This is a Matlab code to generate PCs and ANN regressions for the PCs 
# reaction source terms, diffusion coefficients and other relevant 
# quantities to be used for PC-transport in DNS.
#=====================================================================%
# 
os.chdir("/Users/anujkumar/Documents/RESEARCH/PreProcessingCode/PCA_ANN")
f = open("pcdata.dat","w")
f2= open("pc_ignition.in","w")
f3= open("target_srcvec.dat","w")
f4= open("pc_clookup.dat", "w")
f5= open("QRmat.dat", "w")
f.write("begin_reading_file\n")
#=====================================================================%
#Read the input data
#=====================================================================%
# Positions
xy= np.loadtxt('xy.dat')
# Thermochemical scalars
Y = pd.DataFrame(np.loadtxt('Y.dat'))
# source terms (dT/dt, dY_k/dt)
YDOT = pd.DataFrame(np.loadtxt('YDOT.dat'))
# diffusion coefficients (cgs units)
D = pd.DataFrame(np.loadtxt('D.dat'))
# dynamic viscosity (cgs units)
Visc = pd.DataFrame(np.loadtxt('visc.dat'))
# specific heat ratio
Gamma = pd.DataFrame(np.loadtxt('gamma.dat'))
# molecular weight (cgs units, g/mol)
MolWt = pd.DataFrame(np.loadtxt('molwt.dat'))

# Y min max from the global data
yminmax= pd.Series(np.loadtxt('yminmax.dat'))

# temperature data
Time = Y[0]
Temp = Y[1]
#=====================================================================%
# Prepare the data prior to PCA
#=====================================================================%
# Select temperature and species from which PCA is implemented
#ind = [1   2   3   7     9     10     13     21     24    28]; 
# Selected T:1,H:2,H2:3,H2O:10,CO:13,O2:21, CO2:24, CH3OCH3:28 
TIME  = Y[[0]]
Y     =    Y[[1,2,3,6,7]]
Y.columns = range(Y.shape[1])
YDOT  = YDOT[[1,2,3,6,7]]
YDOT.columns = range(YDOT.shape[1])
D     =    D[[0,1,2,5,6]]
DT    = D.T
DT    = DT.reset_index(drop=True,inplace=False)
D     = DT.T
#Y_min = Y.min()

#Y_max = Y.max()
#Y_rng = Y_max.sub(Y_min)

# Normalize the values of the data for Y prior to carrying out PCA
#min_max_scaler = pp.MinMaxScaler()
#Y_n = pd.DataFrame(2.0*min_max_scaler.fit_transform(Y)-1.0)

Y_max= yminmax[0:5]
Y_min= yminmax[5:10]
Y_max.reset_index(drop=True,inplace=True)
Y_min.reset_index(drop=True,inplace=True)

Y_rng= Y_max-Y_min
Y_avg= Y_max+Y_min

Y_n= 2.0*(Y-Y_min)/Y_rng-1.0
# Generate the corresponding source terms for the normalized Y
YDOT_n = (2.0 * YDOT-Y_avg).div(Y_rng) # this normalization is corrected
# Determine the number of data entries
Y_L = Y.shape[0]
Y_W = Y.shape[1]
#=====================================================================%
# Carry out PCA on the normalized data and determine the required number
# of PCs
#=====================================================================%
C = cov(Y_n.T)
eig_val, Q_matrix = eig(C)
eig_values= -np.sort(-eig_val)
ind= np.argsort(-eig_val)
Q_matrix= Q_matrix[:,ind]
# Compute the PCs vector for the provided data
PCsAll   = pd.DataFrame(Y_n.dot(Q_matrix))
pcDOTAll = pd.DataFrame(YDOT_n.dot(Q_matrix))
N_pc = 5
pc_columns = ['PC_' + str(i+1) for i in range(N_pc)]
PCs = PCsAll[[0,1,2,3,4]]
f2.write("! input parameters for the homogeneous chemistry case with pc transport\n")
f2.write(str(  1) + "        !p_init(atm)\n")
f2.write(str(2000) + "        !T_init(K)\n")
np.savetxt(f2,PCs.iloc[0])
f2.close()

# writing pc_clookup file
f4.write(str(9) + "     ! Total species \n")
f4.write(str(4) + "     ! Selected species \n")
f4.write(str('1 2 5 6\n'))
np.savetxt(f4,np.array(Y_max))
np.savetxt(f4,np.array(Y_min))
#f4.write("%s\n" % Y_min)
# Rotated PCs
# =============================================================================
 Q_red= Q_matrix[:,0:N_pc]
 Rot_mat= varimax(Q_red, 1.0, 20, 1E-8)
# 
 PCs_rot= pd.DataFrame(np.array(PCs).dot(Rot_mat))

# =============================================================================

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    from scipy import eye, asarray, dot, sum
    from scipy.linalg import svd
    from numpy import diag
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        print(i, d/d_old)
        if d_old!=0 and d/d_old < 1 + tol: break
    return R





# Plot the PCs vs. time
ppg = PdfPages('results.pdf')
for i in range(N_pc):
    plt.clf()
    plt.plot(Time*1000,PCs[[i]], c='r', alpha=0.5)
    plt.title('PC # '+str(i+1)+' vs. Time',fontsize=14)
    plt.xlabel('Time (msec)',fontsize=12)
    plt.ylabel('PC ' + str(i+1),fontsize=12)
    unst_plot(xy, PCs[i])
    ppg.savefig()
    plt.show()
    
    
# Plotting the rotated PCs
for i in range(N_pc):
    plt.clf()
    plt.plot(Time*1000,PCs_rot[[i]], c='r', alpha=0.5)
    plt.title('PC # '+str(i+1)+' vs. Time',fontsize=14)
    plt.xlabel('Time (msec)',fontsize=12)
    plt.ylabel('Rotated PC ' + str(i+1),fontsize=12)
    unst_plot(xy, PCs_rot[i])
    #ppg.savefig()
    plt.show()
    
PCs= PCs_rot
    
# Calculate the eigenvalues and plot comulative contribution of the eigenmodes
# on a Scree plot
###variance = pca.explained_variance_ratio_
# Use PCA in sklearn to carry out PCA
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
plt.ylim(90,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(cumul_var,'k-o',markerfacecolor='None',markeredgecolor='k')
ppg.savefig()
#=====================================================================%
# Rotate the PCS to achieve nearly diagonal diffusion coefficients for 
# the PCs

Y_W=N_pc
Rot_mat2= varimax(Q_matrix[:,0:Y_W], 1.0, 20, 1E-8)


QR_matrix= Q_matrix[:,0:Y_W].dot(Rot_mat2)

Dm_store= np.zeros((Y_W, Y_W))
D_array= np.array(D)
D_red= np.zeros((Y_L, Y_W))

for i in range(Y_L):
    Dm_store= (pinv(QR_matrix)).dot(diag(D_array[i,:])).dot(QR_matrix)
    print(Dm_store)
    print((abs(Dm_store).sum()- sum(abs(diag(Dm_store))))/sum(abs(diag(Dm_store))))
    D_red[i,:]= diag(Dm_store)

D_rot= pd.DataFrame(D_red)



#=====================================================================%
pcD = D_rot[[0,1,2,3,4]]

for i in range(N_pc):
# =============================================================================
#     plt.clf()
#     plt.plot(Time*1000,pcD[[i]], c='b', alpha=0.5)
#     plt.title('PC # '+str(i+1)+' Diffusion Coefficient vs. Time',fontsize=14)
#     plt.xlabel('Time (msec)',fontsize=12)
#     plt.ylabel('Diffusion Coefficient' + str(i+1) + '(1/s)',fontsize=12)
# =============================================================================
    unst_plot(xy,pcD[i])
    plt.show()
    
np.savetxt(f4, Rot_mat2.T)
f4.close()

np.savetxt(f5,QR_matrix.T)
f5.close()
#=====================================================================%
# Now recalculate the PCs, their diffusion coefficients and source terms
#=====================================================================%
#Rotaion of the Source Terms
    
pcDOT    = pcDOTAll[[0,1,2,3,4]]
pcDOT= pd.DataFrame(np.array(pcDOT).dot(Rot_mat))

for i in range(N_pc):
# =============================================================================
#     plt.clf()
#     plt.plot(Time*1000,pcDOT[[i]], c='b', alpha=0.5)
#     plt.title('PC # '+str(i+1)+' source term vs. Time',fontsize=14)
#     plt.xlabel('Time (msec)',fontsize=12)
#     plt.ylabel('PC source' + str(i+1) + '(1/s)',fontsize=12)
# =============================================================================
    # ppg.savefig()
    unst_plot(xy, pcDOT[i])
    plt.show()
ppg.close()
#
np.savetxt(f3,pcDOT.iloc[0])
f3.close()
#=====================================================================%
# Carry out ANN regression for source terms, diffusion coefficients and 
# other parameters. In all, the input are the PCs. The ANN architecture
# is different for the different quantities 
#=====================================================================%
# Scale the PCs from -1 to 1
spc_columns = ['SPC_'+ str(i+1) for i in range(N_pc)]
pcScaler = pp.MinMaxScaler()
PCs_n = pd.DataFrame(2*pcScaler.fit_transform(PCs)-1,columns=pc_columns)
srcScaler = pp.MinMaxScaler()
pcDOT_n = pd.DataFrame(2*srcScaler.fit_transform(pcDOT)-1,columns=spc_columns)
src_data = pd.concat([PCs_n,pcDOT_n],axis=1)
srcTrain,srcTest = train_test_split(src_data, test_size = 0.2, random_state= 1)
srcTrain, srcDev = train_test_split(srcTrain, test_size = 0.05, random_state= 2)
srcTrain.reset_index(drop=True,inplace=True)
srcTest.reset_index(drop=True,inplace=True)
srcDev.reset_index(drop=True,inplace=True)
xTrain   = srcTrain[['PC_' + str(i+1) for i in range(N_pc)]]
xTest    = srcTest [['PC_' + str(i+1) for i in range(N_pc)]]
xDev     = srcDev  [['PC_' + str(i+1) for i in range(N_pc)]]
#
input_layer = keras.layers.Input(shape=(N_pc,))

#def custom_activation(x):
 #   return (K.sigmoid(x) * 5) - 1

#model.add(Dense(32 , activation=custom_activation))

layer_1 = keras.layers.Dense(10, activation='tanh')(input_layer)
layer_2 = keras.layers.Dense(10, activation='tanh')(layer_1)
layer_3 = keras.layers.Dense(10, activation='tanh')(layer_2)
layer_4 = keras.layers.Dense(10, activation='tanh')(layer_3)
target  = keras.layers.Dense( 1, activation='tanh')(layer_4)
#
src_model = keras.Model(input_layer,target)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)
src_model.compile(optimizer=optimizer, loss='mse')
#
n_layers  =  4
n_layer_1 = 10
n_layer_2 = 10
n_layer_3 = 10
n_layer_4 = 10
#
pc_max    = pd.DataFrame(PCs.max())
pc_min    = pd.DataFrame(PCs.min())
pcDOT_max = pd.DataFrame(pcDOT.max())
pcDOT_min = pd.DataFrame(pcDOT.min())
#
f.write("%s\n" % N_pc)
f.write("pc_max&min\n")
np.savetxt(f,pc_max)
np.savetxt(f,pc_min)
f.write("source_terms_data\n")
f.write("%s\n" % n_layers)
f.write("%s\n" % n_layer_1)
f.write("%s\n" % n_layer_2)
f.write("%s\n" % n_layer_3)
f.write("%s\n" % n_layer_4)
f.write("source_terms_maxmin\n")
np.savetxt(f,pcDOT_max)
np.savetxt(f,pcDOT_min)
f.write("source_terms_weights\n")
#
n_epochs = 4000
pcvec_n = np.array(PCs_n.loc[0])
for i in range(N_pc):
    f.write("reading_PC" + str(i+1) + "_source_weights\n")
    yTrain = srcTrain[['SPC_'+str(i+1)]]
    yTest  = srcTest [['SPC_'+str(i+1)]]
    yDev   = srcDev  [['SPC_'+str(i+1)]]
    history = src_model.fit(xTrain, yTrain,
                            epochs=n_epochs,
                            shuffle=True,
                            batch_size=128,
                            validation_data=(xDev,yDev),
                            verbose = 0)
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,n_epochs+1)
    plt.semilogy(epochs, loss_train, 'g', label='Training loss')
    plt.semilogy(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss for PC source term # '+str(i),fontsize=14)
    plt.xlabel('Epochs',fontsize=12)
    plt.ylabel('Loss',fontsize=12)
    plt.legend()
    plt.show()
    yPredict = pd.DataFrame(src_model.predict(xTest),columns=['SPC_'+str(i+1)])
    plt.scatter(yTest,yPredict, s = 2, c='r', alpha=0.5)
    plt.title('Model predictions for PC source term # '+str(i+1),fontsize=14)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('Original Source',fontsize=12)
    plt.ylabel('Predicted Source',fontsize=12)
    plt.show()
    corr_coef = yTest.corrwith(yPredict)
    print('The correlation coefficient for PC # ',str(i+1),' is: ',corr_coef)
# Save the weights for the different PCs source terms networks
    sw1 = src_model.layers[1].get_weights()[0].T
    sb1 = src_model.layers[1].get_weights()[1].T
    sw2 = src_model.layers[2].get_weights()[0].T
    sb2 = src_model.layers[2].get_weights()[1].T
    sw3 = src_model.layers[3].get_weights()[0].T
    sb3 = src_model.layers[3].get_weights()[1].T
    sw4 = src_model.layers[4].get_weights()[0].T
    sb4 = src_model.layers[4].get_weights()[1].T
    sw5 = src_model.layers[5].get_weights()[0].T
    sb5 = src_model.layers[5].get_weights()[1].T
   #                    
    for k in range(n_layer_1):
        f.write("%s\n" % sb1[k].astype(np.float64))
    for j in range(N_pc):
        for k in range(n_layer_1):
            f.write("%s\n" % sw1[k,j].astype(np.float64))
    for k in range(n_layer_2):
        f.write("%s\n" % sb2[k].astype(np.float64))
    for j in range(n_layer_1):
        for k in range(n_layer_2):
            f.write("%s\n" % sw2[k,j].astype(np.float64))
    for k in range(n_layer_3):
        f.write("%s\n" % sb3[k].astype(np.float64))
    for j in range(n_layer_2):
        for k in range(n_layer_3):
            f.write("%s\n" % sw3[k,j].astype(np.float64))
    for k in range(n_layer_4):
        f.write("%s\n" % sb4[k].astype(np.float64))
    for j in range(n_layer_3):
        for k in range(n_layer_4):
            f.write("%s\n" % sw4[k,j].astype(np.float64))  
    f.write("%s\n" % sb5[0].astype(np.float64))
    for j in range(n_layer_4):
        f.write("%s\n" % sw5[0,j].astype(np.float64))
  # testing the model predictions
    s1_layer = np.tanh(np.matmul(sw1,pcvec_n)+sb1)
    s2_layer = np.tanh(np.matmul(sw2,s1_layer)+sb2)
    s3_layer = np.tanh(np.matmul(sw3,s2_layer)+sb3)
    s4_layer = np.tanh(np.matmul(sw4,s3_layer)+sb4)
    output = np.tanh(np.matmul(sw5,s4_layer)+sb5)
    srcvec=(output+1.)/2.*(pcDOT_max.loc[i]-pcDOT_min.loc[i])+pcDOT_min.loc[i]
    print('source for PC # ',i+1,' is: ',srcvec[0], 'target = ',pcDOT.loc[0,i])
# =============================================================================

print('Model summary for the PCs source terms: ',src_model.summary())
f.write("end_source_terms_weights\n")
#=====================================================================%
# Carry out ANN regression for diffusion coefficients
#=====================================================================%
f.write("begin_diffcoeff_data\n")
n_layers  = 1
n_layer_1 = 25
f.write("%s\n" % n_layer_1)
f.write("Dc_maxmin\n")
pcD_max = pd.DataFrame(pcD.max())
pcD_min = pd.DataFrame(pcD.min())
np.savetxt(f,pcD_max)
np.savetxt(f,pcD_min)
difScaler = pp.MinMaxScaler()
dpc_columns = ['DPC_' + str(i) for i in range(1, N_pc+1)]
pcD_n = pd.DataFrame(2*difScaler.fit_transform(pcD)-1, columns=dpc_columns)
dif_data = pd.concat([PCs_n,pcD_n],axis=1)
difTrain,difTest = train_test_split(dif_data, test_size = 0.05, random_state= 1)
difTrain, difDev = train_test_split(difTrain, test_size = 0.05, random_state= 2)
difTrain.reset_index(drop=True,inplace=True)
difTest.reset_index(drop=True,inplace=True)
difDev.reset_index(drop=True,inplace=True)
xTrain   = difTrain[['PC_'  + str(i) for i in range(1, N_pc+1)]]
xTest    = difTest [['PC_'  + str(i) for i in range(1, N_pc+1)]]
xDev     = difDev  [['PC_'  + str(i) for i in range(1, N_pc+1)]]
yTrain   = difTrain[['DPC_' + str(i) for i in range(1, N_pc+1)]]
yTest    = difTest [['DPC_' + str(i) for i in range(1, N_pc+1)]]
yDev     = difDev  [['DPC_' + str(i) for i in range(1, N_pc+1)]]
#
input_layer = keras.layers.Input(shape=(N_pc,))
dlayer    = keras.layers.Dense(  25, activation='tanh')(input_layer)
dtarget   = keras.layers.Dense(N_pc, activation='tanh')(dlayer)
dif_model = keras.Model(input_layer,dtarget)
dif_model.compile(optimizer='adam', loss='mse')

n_epochs = 2000
history = dif_model.fit(xTrain, yTrain,
                        epochs=n_epochs,
                        shuffle=True,
                        batch_size=128,
                        validation_data=(xDev,yDev),
                        verbose = 0)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,n_epochs+1)
plt.semilogy(epochs, loss_train, 'g', label='Training loss')
plt.semilogy(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss for PC source term # '+str(i),fontsize=14)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.legend()
plt.show()
yPredict = pd.DataFrame(dif_model.predict(xTest),columns=dpc_columns)
for i in range(1,N_pc+1):
    plt.scatter(yTest[['DPC_'+str(i)]],yPredict[['DPC_'+str(i)]], s = 2, c='b', alpha=0.5)
    plt.title('Model predictions for PC diffusion # '+str(i),fontsize=14)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('Original Source',fontsize=12)
    plt.ylabel('Predicted Source',fontsize=12)
    plt.show()

dcorr_coef = yTest.corrwith(yPredict)
print('The correlation coefficients are: ', dcorr_coef)
print('Model summary for the PCs diffusion terms: ',dif_model.summary())
#
#f.write("\n")
f.write("diff_coeff_weights\n")
sw1d = dif_model.layers[1].get_weights()[0].reshape(N_pc*n_layer_1,1)
sb1d = dif_model.layers[1].get_weights()[1].reshape(     n_layer_1,1)
sw2d = dif_model.layers[2].get_weights()[0].reshape(n_layer_1*N_pc,1)
sb2d = dif_model.layers[2].get_weights()[1].reshape(N_pc          ,1)

np.savetxt(f,sb1d)
np.savetxt(f,sw1d)
np.savetxt(f,sb2d)
np.savetxt(f,sw2d)
f.write("end_diff_weights\n")

#=====================================================================%
# Carry out ANN regression for viscosity
#=====================================================================%
f.write("begin_visccoeff_data\n")
n_layers  = 1
n_layer_1 = 15
f.write("%s\n" % n_layer_1)
f.write("Visc_maxmin\n")
visc_max = pd.DataFrame(Visc.max())
visc_min = pd.DataFrame(Visc.min())
np.savetxt(f,visc_max)
np.savetxt(f,visc_min)
viscScaler = pp.MinMaxScaler()
visc_columns = ['Visc']
visc_n = pd.DataFrame(2*viscScaler.fit_transform(Visc)-1, columns=visc_columns)
visc_data = pd.concat([PCs_n,visc_n],axis=1)
viscTrain,viscTest = train_test_split(visc_data, test_size = 0.2, random_state= 1)
viscTrain, viscDev = train_test_split(viscTrain, test_size = 0.05, random_state= 2)
viscTrain.reset_index(drop=True,inplace=True)
viscTest.reset_index(drop=True,inplace=True)
viscDev.reset_index(drop=True,inplace=True)
xTrain   = viscTrain[['PC_'  + str(i) for i in range(1, N_pc+1)]]
xTest    = viscTest [['PC_'  + str(i) for i in range(1, N_pc+1)]]
xDev     = viscDev  [['PC_'  + str(i) for i in range(1, N_pc+1)]]
yTrain   = viscTrain[['Visc']]
yTest    = viscTest [['Visc']]
yDev     = viscDev  [['Visc']]
#
input_layer = keras.layers.Input(shape=(N_pc,))
vlayer    = keras.layers.Dense(15, activation='tanh')(input_layer)
vtarget   = keras.layers.Dense( 1, activation='tanh')(vlayer)
visc_model = keras.Model(input_layer,vtarget)
visc_model.compile(optimizer='adam', loss='mse')

n_epochs = 2000
history = visc_model.fit(xTrain, yTrain,
                         epochs=n_epochs,
                         shuffle=True,
                         batch_size=128,
                         validation_data=(xDev,yDev),
                         verbose = 0)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,n_epochs+1)
plt.semilogy(epochs, loss_train, 'g', label='Training loss')
plt.semilogy(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss for PC source term # '+str(i),fontsize=14)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.legend()
plt.show()

yPredict = pd.DataFrame(visc_model.predict(xTest),columns=visc_columns)
plt.scatter(yTest[['Visc']],yPredict[['Visc']], c='b', alpha=0.5)
plt.title('Model predictions for viscosity',fontsize=14)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('Original Source',fontsize=12)
plt.ylabel('Predicted Source',fontsize=12)
plt.show()

vcorr_coef = yTest.corrwith(yPredict)
print('The correlation coefficients for viscosity is: ', vcorr_coef)
print('Model summary for the viscosity: ',visc_model.summary())

#f.write("\n")
f.write("visc_coeff_weights\n")
sw1v = visc_model.layers[1].get_weights()[0].reshape(N_pc*n_layer_1,1)
sb1v = visc_model.layers[1].get_weights()[1].reshape(     n_layer_1,1)
sw2v = visc_model.layers[2].get_weights()[0].reshape(n_layer_1,1)
sb2v = visc_model.layers[2].get_weights()[1].reshape(1          ,1)

np.savetxt(f,sb1v)
np.savetxt(f,sw1v)
np.savetxt(f,sb2v)
np.savetxt(f,sw2v)
f.write("end_visc_weights\n")

#=====================================================================%
# Carry out ANN regression for the molecular weight
#=====================================================================%
f.write("begin_molwt_data\n")
n_layers  = 1
n_layer_1 = 15
f.write("%s\n" % n_layer_1)
f.write("molwt_maxmin\n")
molwt_max = pd.DataFrame(MolWt.max())
molwt_min = pd.DataFrame(MolWt.min())
np.savetxt(f,molwt_max)
np.savetxt(f,molwt_min)
#
molwtScaler = pp.MinMaxScaler()
molwt_columns = ['MolWt']
molwt_n = pd.DataFrame(2*molwtScaler.fit_transform(MolWt)-1, columns=molwt_columns)
molwt_data = pd.concat([PCs_n,molwt_n],axis=1)
molwtTrain,molwtTest = train_test_split(molwt_data, test_size = 0.05, random_state= 1)
molwtTrain, molwtDev = train_test_split(molwtTrain, test_size = 0.05, random_state= 2)
molwtTrain.reset_index(drop=True,inplace=True)
molwtTest.reset_index(drop=True,inplace=True)
molwtDev.reset_index(drop=True,inplace=True)
xTrain   = molwtTrain[['PC_'  + str(i) for i in range(1, 6)]]
xTest    = molwtTest [['PC_'  + str(i) for i in range(1, 6)]]
xDev     = molwtDev  [['PC_'  + str(i) for i in range(1, 6)]]
yTrain   = molwtTrain[['MolWt']]
yTest    = molwtTest [['MolWt']]
yDev     = molwtDev  [['MolWt']]
#
input_layer = keras.layers.Input(shape=(N_pc,))
mlayer      = keras.layers.Dense(15, activation='tanh')(input_layer)
mtarget     = keras.layers.Dense( 1, activation='tanh')(mlayer)
molwt_model = keras.Model(input_layer,mtarget)
molwt_model.compile(optimizer='adam', loss='mse')

n_epochs = 2000
history = molwt_model.fit(xTrain, yTrain,
                         epochs=n_epochs,
                         shuffle=True,
                         batch_size=128,
                         validation_data=(xDev,yDev),
                         verbose = 0)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,n_epochs+1)
plt.semilogy(epochs, loss_train, 'g', label='Training loss')
plt.semilogy(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss for PC source term # '+str(i),fontsize=14)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.legend()
plt.show()

yPredict = pd.DataFrame(molwt_model.predict(xTest),columns=molwt_columns)
plt.scatter(yTest[['MolWt']],yPredict[['MolWt']], s = 2, c='b', alpha=0.5)
plt.title('Model predictions for the molecular weight',fontsize=14)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('Original Source',fontsize=12)
plt.ylabel('Predicted Source',fontsize=12)
plt.show()
mcorr_coef = yTest.corrwith(yPredict)
print('The correlation coefficients for the molecular weight is: ', mcorr_coef)
print('Model summary for the molecular weight: ',molwt_model.summary())

#f.write("\n")
f.write("molwt_coeff_weights\n")

sw1m = molwt_model.layers[1].get_weights()[0].reshape(N_pc*n_layer_1,1)
sb1m = molwt_model.layers[1].get_weights()[1].reshape(     n_layer_1,1)
sw2m = molwt_model.layers[2].get_weights()[0].reshape(n_layer_1,1)
sb2m = molwt_model.layers[2].get_weights()[1].reshape(1          ,1)

np.savetxt(f,sb1m)
np.savetxt(f,sw1m)
np.savetxt(f,sb2m)
np.savetxt(f,sw2m)
f.write("end_molwt_weights\n")



#=====================================================================%
# Carry out ANN regression for the specific heat ratio
#=====================================================================%
f.write("begin_gamma_data\n")
n_layers  = 1
n_layer_1 = 15
f.write("%s\n" % n_layer_1)
f.write("gamma_maxmin\n")
gamma_max = pd.DataFrame(Gamma.max())
gamma_min = pd.DataFrame(Gamma.min())
np.savetxt(f,gamma_max)
np.savetxt(f,gamma_min)
#
gammaScaler = pp.MinMaxScaler()
gamma_columns = ['Gamma']
gamma_n = pd.DataFrame(2*gammaScaler.fit_transform(Gamma)-1, columns=gamma_columns)
gamma_data = pd.concat([PCs_n,gamma_n],axis=1)
gammaTrain,gammaTest = train_test_split(gamma_data, test_size = 0.05, random_state= 1)
gammaTrain, gammaDev = train_test_split(gammaTrain, test_size = 0.05, random_state= 2)
gammaTrain.reset_index(drop=True,inplace=True)
gammaTest.reset_index(drop=True,inplace=True)
gammaDev.reset_index(drop=True,inplace=True)
xTrain   = gammaTrain[['PC_'  + str(i) for i in range(1, 6)]]
xTest    = gammaTest [['PC_'  + str(i) for i in range(1, 6)]]
xDev     = gammaDev  [['PC_'  + str(i) for i in range(1, 6)]]
yTrain   = gammaTrain[['Gamma']]
yTest    = gammaTest [['Gamma']]
yDev     = gammaDev  [['Gamma']]
#
input_layer = keras.layers.Input(shape=(N_pc,))
glayer      = keras.layers.Dense(15, activation='tanh')(input_layer)
gtarget     = keras.layers.Dense( 1, activation='tanh')(glayer)
gamma_model = keras.Model(input_layer,gtarget)
gamma_model.compile(optimizer='adam', loss='mse')

n_epochs = 2000
history = gamma_model.fit(xTrain, yTrain,
                         epochs=n_epochs,
                         shuffle=True,
                         batch_size=128,
                         validation_data=(xDev,yDev),
                         verbose = 0)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,n_epochs+1)
plt.semilogy(epochs, loss_train, 'g', label='Training loss')
plt.semilogy(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss for PC source term # '+str(i),fontsize=14)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.legend()
plt.show()

yPredict = pd.DataFrame(gamma_model.predict(xTest),columns=gamma_columns)
plt.scatter(yTest[['Gamma']],yPredict[['Gamma']], s=2, c='b', alpha=0.5)
plt.title('Model predictions for the specific heat ratio',fontsize=14)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('Original Source',fontsize=12)
plt.ylabel('Predicted Source',fontsize=12)
plt.show()
gcorr_coef = yTest.corrwith(yPredict)
print('The correlation coefficient for the specific heat ratio is: ', gcorr_coef)
print('Model summary for the specific heat ratio: ',gamma_model.summary())

#f.write("\n")
f.write("gamma_coeff_weights\n")
sw1g = gamma_model.layers[1].get_weights()[0].reshape(N_pc*n_layer_1,1)
sb1g = gamma_model.layers[1].get_weights()[1].reshape(     n_layer_1,1)
sw2g = gamma_model.layers[2].get_weights()[0].reshape(n_layer_1,1)
sb2g = gamma_model.layers[2].get_weights()[1].reshape(1          ,1)

np.savetxt(f,sb1g)
np.savetxt(f,sw1g)
np.savetxt(f,sb2g)
np.savetxt(f,sw2g)
f.write("end_gamma_weights\n")




#=====================================================================%
# Carry out ANN regression for the Temperature
#=====================================================================%
Temp = pd.DataFrame(Temp)
f.write("begin_Temp_data\n")
n_layers  = 1
n_layer_1 = 15
f.write("%s\n" % n_layer_1)
f.write("Temp_maxmin\n")
temp_max = pd.DataFrame(Temp.max())
temp_min = pd.DataFrame(Temp.min())
np.savetxt(f,temp_max)
np.savetxt(f,temp_min)
#
tempScaler = pp.MinMaxScaler()
temp_columns = ['Temp']
Temp_n = pd.DataFrame(2*tempScaler.fit_transform(Temp)-1, columns=temp_columns)
temp_data = pd.concat([PCs_n,Temp_n],axis=1)
tempTrain,tempTest = train_test_split(temp_data, test_size = 0.05, random_state= 1)
tempTrain, tempDev = train_test_split(tempTrain, test_size = 0.05, random_state= 2)
tempTrain.reset_index(drop=True,inplace=True)
tempTest.reset_index(drop=True,inplace=True)
tempDev.reset_index(drop=True,inplace=True)
xTrain   = tempTrain[['PC_'  + str(i) for i in range(1, 6)]]
xTest    = tempTest [['PC_'  + str(i) for i in range(1, 6)]]
xDev     = tempDev  [['PC_'  + str(i) for i in range(1, 6)]]
yTrain   = tempTrain[['Temp']]
yTest    = tempTest [['Temp']]
yDev     = tempDev  [['Temp']]
#
input_layer = keras.layers.Input(shape=(N_pc,))
tlayer      = keras.layers.Dense(15, activation='tanh')(input_layer)
ttarget     = keras.layers.Dense( 1, activation='tanh')(tlayer)
temp_model = keras.Model(input_layer,ttarget)
temp_model.compile(optimizer='adam', loss='mse')

n_epochs = 2000
history = temp_model.fit(xTrain, yTrain,
                         epochs=n_epochs,
                         shuffle=True,
                         batch_size=128,
                         validation_data=(xDev,yDev),
                         verbose = 0)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,n_epochs+1)
plt.semilogy(epochs, loss_train, 'g', label='Training loss')
plt.semilogy(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss for PC source term # '+str(i),fontsize=14)
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.legend()
plt.show()

yPredict = pd.DataFrame(temp_model.predict(xTest),columns=temp_columns)
plt.scatter(yTest[['Temp']],yPredict[['Temp']], s = 2, c='b', alpha=0.5)
plt.title('Model predictions for the Temperature',fontsize=14)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel('Original Source',fontsize=12)
plt.ylabel('Predicted Source',fontsize=12)
plt.show()
tcorr_coef = yTest.corrwith(yPredict)
print('The correlation coefficients for the Temperature is: ', tcorr_coef)
print('Model summary for the Temperature: ',temp_model.summary())

#f.write("\n")
f.write("temp_coeff_weights\n")

sw1t = temp_model.layers[1].get_weights()[0].reshape(N_pc*n_layer_1,1)
sb1t = temp_model.layers[1].get_weights()[1].reshape(     n_layer_1,1)
sw2t = temp_model.layers[2].get_weights()[0].reshape(n_layer_1,1)
sb2t = temp_model.layers[2].get_weights()[1].reshape(1          ,1)

np.savetxt(f,sb1t)
np.savetxt(f,sw1t)
np.savetxt(f,sb2t)
np.savetxt(f,sw2t)

f.write("end_temp_weights\n")



f.close()



def unst_plot(xy, val):
    xx= xy[:,0]
    yy= xy[:,1]
    
    pval= np.array(val)
    
    plt.tricontourf(xx,yy,pval,20, cmap=plt.cm.jet, extend='both')
    plt.colorbar()
    plt.show
    
    return
    