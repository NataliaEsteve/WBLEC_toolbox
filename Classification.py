#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:46:02 2017

@author: matgilson, ainsabato, vpallares

Modifications by: Natalia Esteve 

"""
import os
import numpy as np
import pandas as pd 
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.svm as sklsvm

os.system('clear')

res_dir = 'classification_results/'
if not os.path.exists(res_dir):
    print('create directory:',res_dir)
    os.makedirs(res_dir)


# DATA AND FEATURE VECTORS 

# data general parameters
n_sub = 18 # number of subjects
n_stage = 4 # awake and sleep stages: 0=awake, 1=n1, 2=n2, 3=n3
N = 90 # number of ROIs

# loading connectivity matrices and masks 
EC = np.load('model_param/J_mod.npy')
mask_EC = np.load('model_param/mask_EC.npy') 
FC = np.load('model_param/FC_emp.npy')
mask_FC = np.load('model_param/mask_FC.npy') #calculated in StatTest_Connectivity
corrFC = np.load('model_param/corrFC.npy')  #calculated in StatTest_Connectivity
mask_corrFC = np.load('model_param/mask_corrFC.npy')  #calculated in StatTest_Connectivity

# Feature Vectors 
vect_EC = EC[:,:,mask_EC] # vectorized EC matrices (only retaining existing connections)
dim_feature_EC= vect_EC.shape[2] # dimension of vectorized EC
vect_FC = FC[:,:,0, mask_FC] 
dim_feature_FC = vect_FC.shape[2] # dimension of vectorized EC
vect_corrFC = corrFC[:,:,mask_corrFC] # vectorized EC matrices (only retaining existing connections)
dim_feature_corrFC = vect_corrFC.shape[2] # dimension of vectorized EC

# z-score Feature Vectors 
z_vect_EC=np.zeros([n_sub,n_stage,dim_feature_EC])
z_vect_FC=np.zeros([n_sub,n_stage,dim_feature_FC])
z_vect_corrFC=np.zeros([n_sub,n_stage,dim_feature_corrFC])
for i_sub in range (n_sub):
    for i_stage in range (n_stage):
        for i in range(dim_feature_EC): 
            z_vect_EC[i_sub,i_stage,i] = (vect_EC[i_sub,i_stage,i]-vect_EC[i_sub,i_stage,:].mean())/vect_EC[i_sub,i_stage,:].std()
        for i in range(dim_feature_FC): 
            z_vect_FC[i_sub,i_stage,i] = (vect_FC[i_sub,i_stage,i]-vect_FC[i_sub,i_stage,:].mean())/vect_FC[i_sub,i_stage,:].std()
        for i in range(dim_feature_corrFC): 
            z_vect_corrFC[i_sub,i_stage,i] = (vect_corrFC[i_sub,i_stage,i]-vect_corrFC[i_sub,i_stage,:].mean())/vect_corrFC[i_sub,i_stage,:].std()
vect_EC[:,:,:]=z_vect_EC[:,:,:]
vect_FC[:,:,:]=z_vect_FC[:,:,:]
vect_corrFC[:,:,:]=z_vect_corrFC[:,:,:]

# labels for classification (train and test sets)
sub_labels = np.repeat(np.arange(n_sub).reshape([-1,1]), n_stage, axis=1)
stage_labels = np.repeat(np.arange(n_stage).reshape([1,-1]), n_sub, axis=0)


# MACHINE LEARNING CLASSIFICATION 

# classifiers and learning parameters
c_MLR_EC = skllm.LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
c_1NN_EC = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
c_SVM_EC = sklsvm.SVC(C=10000, kernel="linear")
c_MLR_FC = skllm.LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
c_1NN_FC = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
c_SVM_FC = sklsvm.SVC(C=10000, kernel="linear")
c_MLR_corrFC = skllm.LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
c_1NN_corrFC = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
c_SVM_corrFC = sklsvm.SVC(C=10000, kernel="linear")

# Classsification performance using subject-wise CV

n_rep = 100  # number of repetition of classification procedure
perf= np.zeros([n_rep,3,3]) # record classification performance for 3 algorithms, and 3 conectivity measurements (0=MLR, 1=1NN and 2=SVM) (0=EC, 1=FC0 and 2=corrFC)  

i=int(n_sub/3*2) # 12 subjects used for training (66.6%) and 6 for testing (33.3%)
# i=int(n_sub/2) # 9 subjects used for training (50%) and 9 for testing (50%)

# perform classification
for i_rep in range(n_rep):
    
    # split run indices in train and test sets
    sub_train_labels = np.zeros([n_sub],dtype=bool)
    sub_test_labels = np.ones([n_sub],dtype=bool)
    while sub_train_labels.sum()<i:
        rnd_int = np.random.randint(n_sub)
        if not sub_train_labels[rnd_int]:
            sub_train_labels[rnd_int] = True
            sub_test_labels[rnd_int] = False
    
    # train and test classifiers with subject labels # perf (0=MLR, 1=1NN and 2=SVM) (0=EC, 1=FC0 and 2=corrFC)
    c_MLR_EC.fit(vect_EC[sub_train_labels,:,:].reshape([-1,dim_feature_EC]), stage_labels[sub_train_labels,:].reshape([-1]))
    c_MLR_FC.fit(vect_FC[sub_train_labels,:,:].reshape([-1,dim_feature_FC]), stage_labels[sub_train_labels,:].reshape([-1]))
    c_MLR_corrFC.fit(vect_corrFC[sub_train_labels,:,:].reshape([-1,dim_feature_corrFC]), stage_labels[sub_train_labels,:].reshape([-1]))
    perf[i_rep,0,0] = c_MLR_EC.score(vect_EC[sub_test_labels,:,:].reshape([-1,dim_feature_EC]), stage_labels[sub_test_labels,:].reshape([-1]))
    perf[i_rep,0,1] = c_MLR_FC.score(vect_FC[sub_test_labels,:,:].reshape([-1,dim_feature_FC]), stage_labels[sub_test_labels,:].reshape([-1]))
    perf[i_rep,0,2] = c_MLR_corrFC.score(vect_corrFC[sub_test_labels,:,:].reshape([-1,dim_feature_corrFC]), stage_labels[sub_test_labels,:].reshape([-1]))

    c_1NN_EC.fit(vect_EC[sub_train_labels,:,:].reshape([-1,dim_feature_EC]), stage_labels[sub_train_labels,:].reshape([-1]))
    c_1NN_FC.fit(vect_FC[sub_train_labels,:,:].reshape([-1,dim_feature_FC]), stage_labels[sub_train_labels,:].reshape([-1]))
    c_1NN_corrFC.fit(vect_corrFC[sub_train_labels,:,:].reshape([-1,dim_feature_corrFC]), stage_labels[sub_train_labels,:].reshape([-1]))
    perf[i_rep,1,0] = c_1NN_EC.score(vect_EC[sub_test_labels,:,:].reshape([-1,dim_feature_EC]), stage_labels[sub_test_labels,:].reshape([-1]))
    perf[i_rep,1,1] = c_1NN_FC.score(vect_FC[sub_test_labels,:,:].reshape([-1,dim_feature_FC]), stage_labels[sub_test_labels,:].reshape([-1]))
    perf[i_rep,1,2] = c_1NN_corrFC.score(vect_corrFC[sub_test_labels,:,:].reshape([-1,dim_feature_corrFC]), stage_labels[sub_test_labels,:].reshape([-1]))

    c_SVM_EC.fit(vect_EC[sub_train_labels,:,:].reshape([-1,dim_feature_EC]), stage_labels[sub_train_labels,:].reshape([-1]))
    c_SVM_FC.fit(vect_FC[sub_train_labels,:,:].reshape([-1,dim_feature_FC]), stage_labels[sub_train_labels,:].reshape([-1]))
    c_SVM_corrFC.fit(vect_corrFC[sub_train_labels,:,:].reshape([-1,dim_feature_corrFC]), stage_labels[sub_train_labels,:].reshape([-1]))
    perf[i_rep,2,0] = c_SVM_EC.score(vect_EC[sub_test_labels,:,:].reshape([-1,dim_feature_EC]), stage_labels[sub_test_labels,:].reshape([-1]))
    perf[i_rep,2,1] = c_SVM_FC.score(vect_FC[sub_test_labels,:,:].reshape([-1,dim_feature_FC]), stage_labels[sub_test_labels,:].reshape([-1]))
    perf[i_rep,2,2] = c_SVM_corrFC.score(vect_corrFC[sub_test_labels,:,:].reshape([-1,dim_feature_corrFC]), stage_labels[sub_test_labels,:].reshape([-1]))

# Performance dataframes 
# Classification performance dataframe: mean accuracy of n_rep training and testing procedure, for each classifier    
perfdic={'MLR with EC':perf[:,0,0],'MLR with FC':perf[:,0,1],'MLR with corrFC':perf[:,0,2],
      '1NN with EC':perf[:,1,0],'1NN with FC':perf[:,1,1],'1NN with corrFC':perf[:,1,2],
      'SVM with EC':perf[:,2,0],'SVM with FC':perf[:,2,1],'SVM with corrFC':perf[:,2,2]}
perfdf=pd.DataFrame(data=perfdic)

# Mean classification performance dataframe: Mean accuracy and standar deviation for each classifier 
perfdic_mean={'MLR':[perf[:,0,0].mean(),perf[:,0,0].std(),perf[:,0,1].mean(),perf[:,0,1].std(),perf[:,0,2].mean(),perf[:,0,2].std()],
              '1NN':[perf[:,1,0].mean(),perf[:,1,0].std(),perf[:,1,1].mean(),perf[:,1,1].std(),perf[:,1,2].mean(),perf[:,1,2].std()],
              'SVM':[perf[:,2,0].mean(),perf[:,2,0].std(),perf[:,2,1].mean(),perf[:,2,1].std(),perf[:,2,2].mean(),perf[:,2,2].std()]}
perfdf_mean=pd.DataFrame(data=perfdic_mean, index=['EC mean','EC stdv','FC mean','FC stdv','corrFC mean','corrFC stdv'] )
perfdf_mean_round=perfdf_mean.round(2) #rounding data to nearest hundredths

perfarray=[[perf[:,0,0].mean(),perf[:,0,0].std(),perf[:,0,1].mean(),perf[:,0,1].std(),perf[:,0,2].mean(),perf[:,0,2].std()],
          [perf[:,1,0].mean(),perf[:,1,0].std(),perf[:,1,1].mean(),perf[:,1,1].std(),perf[:,1,2].mean(),perf[:,1,2].std()],
          [perf[:,2,0].mean(),perf[:,2,0].std(),perf[:,2,1].mean(),perf[:,2,1].std(),perf[:,2,2].mean(),perf[:,2,2].std()]]
perfdf_mean_2=pd.DataFrame( data=perfarray,index= ['MLR','1NN', 'SVM'],columns= ['EC mean','EC stdv','FC mean','FC stdv','corrFC mean','corrFC stdv'])
perfdf_mean_round_2=perfdf_mean_2.round(2) #rounding data to nearest hundredths


# VISUALIZING AND SAVING RESULTS 

import matplotlib.pyplot as plt
import seaborn as sns

suffix="_train66_" #suffix for reference of training vs testing group size when saving results 
n="2" #used to index figures when saving
#plt.style.available
plt.style.use('seaborn-colorblind') # use the 'seaborn-colorblind' style for plotting 

#save classification perfoprmance and mean classification performance dataframe as csv
perfdf_mean_round.to_csv(res_dir+"classification_perforfance_mean"+suffix+n+".csv")
perfdf_mean_round_2.to_csv(res_dir+"classification_perforfance_mean2"+suffix+n+".csv")
perfdf.to_csv(res_dir+"classification_perforfance"+suffix+n+".csv")
#p=pd.read_csv(res_dir+"classification_perforfance"+suffix+n+".csv")

# classification performance plots  
perfdf.plot();
perfdf.plot.box();
perfdf.plot.hist(alpha=0.7); #histogram 
plt.savefig(res_dir+"classification_perforfance_histogram"+suffix+n+".pdf")
perfdf.plot.kde(); #kernel density estimation plot 
plt.savefig(res_dir+"classification_perforfance_kernel"+suffix+n+".pdf")
sns.violinplot(data=perfdf) #violinplots all classifiers with all features 
plt.savefig(res_dir+"classification_perforfance_violinplot"+suffix+n+".pdf")
sns.violinplot(data=perfdf[['1NN with EC','1NN with FC','1NN with corrFC']]) #vilinplots of 1NN classifiers
plt.savefig(res_dir+"classification_perforfance_violinplot_1NN"+suffix+n+".pdf")
sns.violinplot(data=perfdf[['MLR with EC','MLR with FC','MLR with corrFC']]) #vilinplots of MLR classifiers
plt.savefig(res_dir+"classification_perforfance_violinplot_MLR"+suffix+n+".pdf")
sns.violinplot(data=perfdf[['SVM with EC','SVM with FC','SVM with corrFC']]) #vilinplots of SVM classifiers
plt.savefig(res_dir+"classification_perforfance_violinplot_SVM"+suffix+n+".pdf")
sns.violinplot(data=perfdf[['SVM with EC','MLR with EC','1NN with EC']]) #vilinplots classifiers coupled with EC
plt.savefig(res_dir+"classification_perforfance_violinplot_EC"+suffix+n+".pdf")
sns.violinplot(data=perfdf[['SVM with corrFC','MLR with corrFC','1NN with corrFC']]) #vilinplots classifiers coupled with corrFC
plt.savefig(res_dir+"classification_perforfance_violinplot_corrFC"+suffix+n+".pdf")


# mean classification performance plots 

perfdf_mean_round_2[["EC mean","FC mean","corrFC mean"]].plot();
plt.savefig(res_dir+"classification_perforfance_mean2_lineplot"+suffix+n+".pdf")
perfdf_mean_round_2[["EC mean","FC mean","corrFC mean"]].plot.bar(yerr=[perfdf_mean_round_2["EC stdv"],perfdf_mean_round_2["FC stdv"],perfdf_mean_round_2["corrFC stdv"]])
plt.savefig(res_dir+"classification_perforfance_mean2_barchart"+suffix+n+".pdf")

# classification performance plots: multiple violin plots (no sns)

plt.violinplot((perf[:,0,0],perf[:,2,2],perf[:,0,1]),positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC with MLR',"corrFC with SVM","FC with MLR" ],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Performance Comparison',fontsize=16)
plt.show()     

plt.violinplot(perf[:,0,:],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC','FC0', "corrFC"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('MLR Classification Performance',fontsize=16)
plt.show()

plt.violinplot(perf[:,1,:],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC','FC0', "corrFC"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('1NN Classification Performance',fontsize=16)
plt.show()

plt.violinplot(perf[:,2,:],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC','FC0', "corrFC"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('SVM Classification Performance',fontsize=16)
plt.show()


plt.violinplot(perf[:,:,0],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['MLR','1NN', "SVM"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Classification Performance for EC ',fontsize=16)
plt.show()

plt.violinplot(perf[:,:,1],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['MLR','1NN', "SVM"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Classification Performance for FC ',fontsize=16)
plt.show()

plt.violinplot(perf[:,:,2],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['MLR','1NN', "SVM"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Classification Performance for corrFC ',fontsize=16)
plt.show()


# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split

i=int(n_sub/4)
i=int(n_sub/3*2)
for i_rep in range(n_rep):
    # split run indices in train and test sets
    sub_train_labels = np.zeros([n_sub],dtype=bool)
    sub_test_labels = np.ones([n_sub],dtype=bool)
    while sub_train_labels.sum()<i:
        rnd_int = np.random.randint(n_sub)
        if not sub_train_labels[rnd_int]:
            sub_train_labels[rnd_int] = True
            sub_test_labels[rnd_int] = False
    #print('train/test sets:',sub_train_labels,sub_test_labels)
    
y_true=stage_labels[sub_test_labels,:].reshape([-1])
y_pred=c_MLR_EC.predict(vect_EC[sub_test_labels,:,:].reshape([-1,dim_feature_EC]))
print(y_true.shape)
print(y_pred.shape)

CM = confusion_matrix(y_true=y_true, y_pred=y_pred)
plt.figure()
plt.imshow(CM, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.linspace(0,n_stage-1,num=n_stage),('awake', 'N1', 'N2', 'N3'),fontsize=8)
plt.yticks(np.linspace(0,n_stage-1,num=n_stage),('awake', 'N1', 'N2', 'N3'),fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('MLR Confusion Matrix for EC',fontsize=12)
plt.show()


CM=np.zeros([n_stage, n_stage])
i=int(n_sub/2)
n_rep=2
for i_rep in range(n_rep):
    # split run indices in train and test sets
    sub_train_labels = np.zeros([n_sub],dtype=bool)
    sub_test_labels = np.ones([n_sub],dtype=bool)
    while sub_train_labels.sum()<i:
        rnd_int = np.random.randint(n_sub)
        if not sub_train_labels[rnd_int]:
            sub_train_labels[rnd_int] = True
            sub_test_labels[rnd_int] = False
    #print('train/test sets:',sub_train_labels,sub_test_labels)
    y_true=stage_labels[sub_test_labels,:].reshape([-1])
    y_pred=c_MLR_EC.predict(vect_EC[sub_test_labels,:,:].reshape([-1,dim_feature_EC]))
    print(y_true.shape)
    print(y_pred.shape)
    CM_temp = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(CM_temp)
    CM=CM+CM_temp 

CM = confusion_matrix(y_true=y_true, y_pred=y_pred)
plt.figure()
plt.imshow(CM, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.linspace(0,n_stage-1,num=n_stage),('awake', 'N1', 'N2', 'N3'),fontsize=8)
plt.yticks(np.linspace(0,n_stage-1,num=n_stage),('awake', 'N1', 'N2', 'N3'),fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('MLR Confusion Matrix for EC',fontsize=12)
plt.show()