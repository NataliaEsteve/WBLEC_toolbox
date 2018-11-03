#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:46:02 2017

@author: matgilson, ainsabato, vpallares

Modifications by: Natalia Esteve 

"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.svm as sklsvm

# In this script we:
# 1. Calculate EC, FC0 and corrFC z-score feature vectors  
# 2. Perform clasification using MLR, 1NN and SMV, coupled with EC, FC0 and corrFC feature vectors  
# 3. Calculate Confusion Matrix for MLR 

# fMRI properties
n_sub = 18 # number of subjects
n_stage = 4 # awake and sleep stages: 0=awake, 1=n1, 2=n2, 3=n3
N = 90 # number of ROIs

# loading connectivity matrices and masks 
EC = np.load('model_param/J_mod.npy') #calculated in ParameterEstimation.py
mask_EC = np.load('model_param/mask_EC.npy') #calculated in ParameterEstimation.py
FC = np.load('model_param/FC_emp.npy') #calculated in ParameterEstimation.py 
mask_FC = np.load('model_param/mask_FC.npy') #calculated in SimilarityAnalysis.py  
corrFC = np.load('model_param/corrFC.npy')  #calculated in SimilarityAnalysis.py    
mask_corrFC = np.load('model_param/mask_corrFC.npy')  #calculated in SimilarityAnalysis.py    

# feature vectors 
vect_EC = EC[:,:,mask_EC] # vectorized EC matrices (only retaining existing connections)
dim_feature_EC= vect_EC.shape[2] # dimension of vectorized EC
vect_FC = FC[:,:,0, mask_FC] 
dim_feature_FC = vect_FC.shape[2] # dimension of vectorized EC
vect_corrFC = corrFC[:,:,mask_corrFC] # vectorized EC matrices (only retaining existing connections)
dim_feature_corrFC = vect_corrFC.shape[2] # dimension of vectorized EC

# z-score feature vectors 
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

# labels for classification (train+test)
sub_labels = np.repeat(np.arange(n_sub).reshape([-1,1]), n_stage, axis=1)
stage_labels = np.repeat(np.arange(n_stage).reshape([1,-1]), n_sub, axis=0)

# classifier and learning parameters
c_MLR_EC = skllm.LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
c_1NN_EC = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
c_SVM_EC = sklsvm.SVC(C=10000, kernel="linear")
c_MLR_FC = skllm.LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
c_1NN_FC = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
c_SVM_FC = sklsvm.SVC(C=10000, kernel="linear")
c_MLR_corrFC = skllm.LogisticRegression(C=10000, penalty='l2', multi_class='multinomial', solver='lbfgs')
c_1NN_corrFC = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
c_SVM_corrFC = sklsvm.SVC(C=10000, kernel="linear")

# perform classification
n_rep = 100  # number of repetition of classification procedure
perf= np.zeros([n_rep,3,3]) # record classification performance for 3 algorithms, and 3 conectivity measurements (0=MLR, 1=1NN and 2=SVM) (0=EC, 1=FC0 and 2=corrFC)  

i=int(n_sub/3*2) # 12 subjects used for training (66.6%) and 6 for testing (33.3%)
# i=int(n_sub/2) # 9 subjects used for training (50%) and 9 for testing (50%)
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
    
    # train and test classifiers with subject labels
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


# Classifiers performance (accuracy)
print("MLR Performance for EC, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,0,0].mean(), perf[:,0,0].std(), np.median(perf[:,0,0]))) 
print("MLR Performance for FC0, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,0,1].mean(), perf[:,0,1].std(), np.median(perf[:,0,1]))) 
print("MLR Performance for corrFC, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,0,2].mean(), perf[:,0,2].std(), np.median(perf[:,0,2]))) 
print("1NN Performance for EC, Accuracy: %1.2f +/- %.2f (mean +/- std, %1.2f median" %(perf[:,1,0].mean(), perf[:,1,0].std(), np.median(perf[:,1,0]))) 
print("1NN Performance for FC0, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,1,1].mean(), perf[:,1,1].std(), np.median(perf[:,1,1]))) 
print("1NN Performance for corrFC, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,1,2].mean(), perf[:,1,2].std(), np.median(perf[:,1,2]))) 
print("SVM Performance for EC, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,2,0].mean(), perf[:,2,0].std(), np.median(perf[:,2,0]))) 
print("SVM Performance for FC0, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,2,1].mean(), perf[:,2,1].std(), np.median(perf[:,2,1]))) 
print("SVM Performance for corrFC, Accuracy: %1.2f +/- %.2f (mean +/- std), %1.2f median" %(perf[:,2,2].mean(), perf[:,2,2].std(), np.median(perf[:,2,2]))) 

perf_dataframe=[['Algorithm','EC mean','EC stdv','FC mean','FC stdv','corrFC mean','corrFC stdv'],
                      ['MLR',perf[:,0,0].mean(),perf[:,0,0].std(),perf[:,0,1].mean(),perf[:,0,1].std(),perf[:,0,2].mean(),perf[:,0,2].std()],
                      ['1NN',perf[:,1,0].mean(),perf[:,1,0].std(),perf[:,1,1].mean(),perf[:,1,1].std(),perf[:,1,2].mean(),perf[:,1,2].std()],
                      ['SVM',perf[:,2,0].mean(),perf[:,2,0].std(),perf[:,2,1].mean(),perf[:,2,1].std(),perf[:,2,2].mean(),perf[:,2,2].std()]]

 
# violin plots comparing classifiers performance (accuracy)

n="0" #used to index figures when saving

# Violin plot comparing performance between the classifiers EC coupled MLR v.s. corrFC with SVM v.s. FC with MLR 
plt.violinplot((perf[:,0,0],perf[:,2,2],perf[:,0,1]),positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC with MLR',"corrFC with SVM","FC with MLR" ],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Performance Comparison',fontsize=16)
plt.savefig('Performance Comparison'+'_'+n+'.pdf')
plt.show() 

# Violin plot of comparing performance of an algorithm coupled with different connectivity measures (EC v.s. corrFC v.s. FC) 
plt.violinplot(perf[:,0,:],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC','FC0', "corrFC"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('MLR Classification Performance',fontsize=16)
plt.savefig('MLR Classification Performance'+'_'+n+'.pdf')
plt.show()

plt.violinplot(perf[:,1,:],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC','FC0', "corrFC"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('1NN Classification Performance',fontsize=16)
plt.savefig('1NN Classification Performance'+'_'+n+'.pdf')
plt.show()

plt.violinplot(perf[:,2,:],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['EC','FC0', "corrFC"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('SVM Classification Performance',fontsize=16)
plt.savefig('SVM Classification Performance'+'_'+n+'.pdf')
plt.show()

# Violin plot of comparing performance a connectivity measure coupled with different algorithms (MLR v.s. 1NN v.s. SMV)
plt.violinplot(perf[:,:,0],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['MLR','1NN', "SVM"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Classification Performance for EC',fontsize=16)
plt.savefig('Classification Performance for EC'+'_'+n+'.pdf')
plt.show()

plt.violinplot(perf[:,:,1],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['MLR','1NN', "SVM"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Classification Performance for FC ',fontsize=16)
plt.savefig('Classification Performance for FC'+'_'+n+'.pdf')
plt.show()

plt.violinplot(perf[:,:,2],positions=[0,1,2],showmeans=True)
plt.axis(xmin=-0.4,xmax=2.4,ymin=0.25,ymax=1)
plt.xticks([0,1,2],['MLR','1NN', "SVM"],fontsize=10)
plt.ylabel('accuracy',fontsize=10)
plt.title('Classification Performance for corrFC',fontsize=16)
plt.savefig('Classification Performance for corrFC'+'_'+n+'.pdf')
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