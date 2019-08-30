# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:59:26 2019
%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the sparse representation 
% of each data point in terms of the rest of the points
% X: DxN data matrix
% Y = YC 
% affine: if true then enforce the affine constraint
% thr1: stopping threshold for the coefficient error ||Z-C||
% thr2: stopping threshold for the linear system error ||Y-YZ||
% maxIter: maximum number of iterations of ADMM
% C2: NxN sparse coefficient matrix
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------
@ fanmingyu@wzu.edu.cn  rewritten this code from matlab to python simply using numpy.
"""

import numpy as np

class ssc_model(object):
    def __init__(self, X, affine=False, alpha1=800,  alpha2 = None, thr=0.0002, maxIter=200):
        self.alpha1 = alpha1 
        if not alpha2:
            self.alpha2 = alpha1
        else:
            self.alpha2 = alpha2
        
        self.X = X
        self.affine = affine    
        self.thr = thr
        self.maxIter = maxIter
        self.N = X.shape[1]   # number of samples
        
        self.T = np.dot(self.X.T,self.X)
        T1 = np.abs(self.T - np.diag(np.diag(self.T)))
        self.lambda1 = np.min(np.max(T1,axis=1))
        self.mu1 = self.alpha1/self.lambda1
        self.mu2 = self.alpha2 
        self.I = np.eye(self.N,dtype=np.float32)
        self.ones = np.ones((self.N,self.N),dtype=np.float32)
        self.vec1N = np.ones((1,self.N),dtype = np.float32)
        self.err =[]
        
    def computeCmat(self):
        if not self.affine:
            A = np.linalg.inv(self.mu1*self.T + self.mu2*self.I)
            C1 = np.zeros((self.N,self.N),dtype=np.float32)
            Lambda2 = np.zeros((self.N,self.N),dtype=np.float32)
            err = 10*self.thr
            iter1 = 1
            while (err>self.thr)and(iter1<self.maxIter):
                #update Z
                Z = np.dot(A,self.mu1*self.T + self.mu2*(C1 - Lambda2/self.mu2))
                Z = Z - np.diag(np.diag(Z))
                # update C
                tmp_val = np.abs(Z + Lambda2/self.mu2) - (self.ones/self.mu2)
                C2 = np.maximum(0,tmp_val)*np.sign(Z + Lambda2/self.mu2)
                C2 = C2 - np.diag(np.diag(C2))
                # update lagrangian multipliers
                Lambda2 = Lambda2 + self.mu2*(Z-C2)
                # compute errors
                tmp_val = np.abs(Z - C2)
                err = np.max(tmp_val.reshape(-1,1))
                C1 = C2
                iter1 = iter1 +1
                print('the error is = %f' % err)
        else:
            A = np.linalg.inv(self.mu1*self.T + self.mu2*self.I+ self.mu2*self.ones)
            C1 = np.zeros((self.N,self.N),dtype=np.float32)
            Lambda2 = np.zeros((self.N,self.N),dtype=np.float32)
            Lambda3 = np.zeros((1,self.N),dtype=np.float32)
            err1 = 10*self.thr
            err3 = 10*self.thr
            iter1 = 1
            while ((err1>self.thr)or(err3>self.thr))and(iter1<self.maxIter):
                #update Z
                tmp_val = self.mu1*self.T + self.mu2*(C1-Lambda2/self.mu2) + self.mu2*np.dot(self.vec1N.T,(self.vec1N - Lambda3/self.mu2))
                Z = np.dot(A,tmp_val)
                Z = Z - np.diag(np.diag(Z))
                # update C
                tmp_val = np.abs(Z + Lambda2/self.mu2) - (self.ones/self.mu2)
                C2 = np.maximum(0,tmp_val)*np.sign(Z + Lambda2/self.mu2)
                C2 = C2 - np.diag(np.diag(C2))
                # update lagrangian multipliers
                Lambda2 = Lambda2 + self.mu2*(Z-C2)
                Lambda3 = Lambda3 + self.mu2*(np.dot(self.vec1N,Z) - self.vec1N)
                # compute errors
                tmp_val = np.abs(Z - C2)
                err1 = np.max(tmp_val.reshape(-1,1))
                tmp_val = np.abs(np.dot(self.vec1N,Z) - self.vec1N)
                err3 = np.max(tmp_val.reshape(-1,1))
                
                C1 = C2
                iter1 = iter1 + 1
                print('iter1 = %d, the error 1 is = %f and error 2 is %f' % (iter1, err1, err3))
        return C2
