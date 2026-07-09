#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:40:29 2020

@author: ying
"""

import os
from bsr.funcs import Operator, Node
from bsr.funcs import grow, genList, shrink, upgOd, allcal, display, getHeight, getNum, numLT, upDepth, Express, fStruc
from bsr.funcs import ylogLike, newProp, Prop, auxProp


import numpy as np
import pandas as pd
from scipy.stats import invgamma
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
import sklearn
import copy
import matplotlib.pyplot as plt
import random
import time
from joblib import Parallel, delayed

OPERATION_SETS = {
    #extended set used in the original repo
    "default": {
        "ops": ['inv', 'ln', 'neg', 'sin', 'cos', 'exp', 'square', 'cubic', '+', '*'],
        "types": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
    },
    #original paper set
    "paper": {
        "ops": ['inv', 'ln', 'neg', 'exp', '+', '*'],
        "types": [1, 1, 1, 1, 2, 2],
    },
    # EML set
    "exp_log": {
        "ops": ['exp_minus_log'],
        "types": [2],
        "constant_terminal": True,
        "max_complexity": 500,
        "max_depth": 12,
        "annealing": True,
    },
}


def get_operation_set(name):
    if name not in OPERATION_SETS:
        valid = ", ".join(sorted(OPERATION_SETS))
        raise ValueError(f"Unknown operation set {name!r}. Valid operation sets: {valid}")
    operation_set = OPERATION_SETS[name]
    ops = operation_set["ops"]
    weights = operation_set.get("weights", [1.0 / len(ops)] * len(ops))
    return (
        ops,
        weights,
        operation_set["types"],
        operation_set.get("constant_terminal", False),
        operation_set.get("max_complexity"),
        operation_set.get("max_depth"),
    )


class BSR(BaseEstimator,RegressorMixin):
    def __init__(self, treeNum=3, itrNum=5000, alpha1 = 0.4, alpha2=0.4,  
                 beta=-1, disp=False, val=100, operation_set="default", max_complexity=None, max_depth=None, tree_dtype=float, n_jobs=1,
                 T_start=None, cool_fraction=None, left_prior={"exp_minus_log": {1.0: 0.5}}, right_prior=None, eml_unary_chain_prior=None, num_cycles=3):
        self.treeNum = treeNum
        self.itrNum = itrNum
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta #WGL
        self.disp = disp #WGL
        self.val = val #WGL
        self.operation_set = operation_set
        self.max_complexity = max_complexity
        self.max_depth = max_depth
        self.tree_dtype = tree_dtype
        self.n_jobs = n_jobs
        self.T_start = T_start
        self.cool_fraction = cool_fraction
        self.left_prior = left_prior
        self.right_prior = right_prior
        self.num_cycles = num_cycles
        if eml_unary_chain_prior is None:
            self.eml_unary_chain_prior = True if operation_set == "exp_log" else False
        else:
            self.eml_unary_chain_prior = eml_unary_chain_prior
        
    def model(self, last_ind=1):
        modd =[]
        for i in  range(self.treeNum):
            if np.dtype(self.tree_dtype).kind == 'c':
                modd.append(f"Re({Express(self.roots_[-last_ind][i])})")
            else:
                modd.append(Express(self.roots_[-last_ind][i]))
        return(modd)
            
    def complexity(self):
        compl = 0
        cmpls = []
        for i in  range(self.treeNum):
            root_node = self.roots_[-1][i]
            numm = getNum(root_node)
            cmpls.append(numm)
            compl = compl + numm
        return(compl)
        
    def predict(self, test_data, method = 'last', last_ind = 1):
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data, dtype=self.tree_dtype)
        else:
            test_data = test_data.astype(self.tree_dtype)
        K = self.treeNum
        n_test = test_data.shape[0]
        XX = np.zeros((n_test, K), dtype=self.tree_dtype)
        if method == 'last':
            for countt in np.arange(K):
                temp = allcal(self.roots_[-last_ind][countt], test_data)
                temp.shape = (temp.shape[0])
                XX[:, countt] = temp
            if np.iscomplexobj(XX):
                XX = np.real(XX)
            constant = np.ones((n_test, 1))
            XX = np.concatenate((constant, XX), axis=1)
            Beta = self.betas_[-last_ind]
            toutput = np.matmul(XX, Beta)
        return(toutput)
    
    def _run_one_iteration(self, train_data, train_y, n_feature, n_train):
        K = self.treeNum
        beta = self.beta
        
        Ops, Op_weights, Op_type, const_terminal, default_max_complexity, default_max_depth = get_operation_set(self.operation_set)
        max_complexity = self.max_complexity if self.max_complexity is not None else default_max_complexity
        max_depth = self.max_depth if self.max_depth is not None else default_max_depth
        
        # Determine whether to use annealing based on the operation set config
        operation_set_config = OPERATION_SETS.get(self.operation_set, {})
        use_annealing = operation_set_config.get("annealing", False)
        T_start = self.T_start if self.T_start is not None else (50.0 if use_annealing else 1.0)
        cool_fraction = self.cool_fraction if self.cool_fraction is not None else 0.7
        cool_steps = int(self.val * cool_fraction)
        
        RootLists = []
        for i in np.arange(K):
            RootLists.append([])
        
        SigaList = []
        SigbList = []
        sigma = invgamma.rvs(1)
        
        for count in np.arange(K):
            Root = Node(0)
            Root.eml_unary_chain_prior = self.eml_unary_chain_prior
            sigma_a = invgamma.rvs(1)
            sigma_b = invgamma.rvs(1)
        
            if self.disp: print('grow a tree from the Root node')
            other_complexity = 0
            for prev_count in range(count):
                other_complexity += getNum(RootLists[prev_count][-1])
            
            budget = max_complexity - other_complexity if max_complexity is not None else None
            grow(Root, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b, const_terminal, max_depth, max_complexity=budget, node_count=[1], left_prior=self.left_prior, right_prior=self.right_prior)
        
            RootLists[count].append(copy.deepcopy(Root))
            SigaList.append(sigma_a)
            SigbList.append(sigma_b)
        
        if self.disp: print('calculate beta')
        XX = np.zeros((n_train, K), dtype=self.tree_dtype)
        for count in np.arange(K):
            temp = allcal(RootLists[count][-1], train_data)
            temp.shape = (temp.shape[0])
            XX[:, count] = temp
            
        if np.iscomplexobj(XX):
            XX = np.real(XX)
            
        constant = np.ones((n_train,1))
        XX = np.concatenate((constant, XX), axis=1)
        scale = np.max(np.abs(XX), axis=0)
        scale[scale == 0] = 1.0
        XX = XX / scale
        epsilon = np.eye(XX.shape[1])*1e-6
        yy = np.array(train_y)
        yy.shape = (yy.shape[0], 1)
        Beta = np.linalg.inv(np.matmul(XX.transpose(), XX)+epsilon)
        Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))
        output = np.matmul(XX, Beta)
        Beta = Beta / scale.reshape(-1, 1)
        
        total = 0
        accepted = 0
        errList = []
        totList = []
        nodeCounts = []
        
        tic = time.time()
        
        if self.disp: print('while total < ',self.val)
        
        cycle_length = max(1, int(cool_steps / self.num_cycles)) if hasattr(self, 'num_cycles') else max(1, int(cool_steps / 3))
        
        while total < self.val:
            if use_annealing and total < cool_steps:
                cycle_progress = (total % cycle_length) / cycle_length
                temperature = 1.0 + 0.5 * (T_start - 1.0) * (1 + np.cos(np.pi * cycle_progress))
            else:
                temperature = 1.0

            switch_label = False
            for count in np.arange(K):
                Roots = []
                for ccount in np.arange(K):
                    Roots.append(RootLists[ccount][-1])
                sigma_a = SigaList[count]
                sigma_b = SigbList[count]
        
                if self.disp: print(f'newProp (temp={temperature:.3f})...')
                [res, sigma, Root, sigma_a, sigma_b] = newProp(
                    Roots, count, sigma, train_y, train_data, n_feature, Ops,
                    Op_weights, Op_type, beta, sigma_a, sigma_b,
                    const_terminal, max_complexity, max_depth,
                    temperature=temperature,
                    left_prior=self.left_prior,
                    right_prior=self.right_prior
                )
                if self.disp:
                    print("res:",res)
                    display(genList(Root))
        
                total += 1
                SigaList[count] = sigma_a
                SigbList[count] = sigma_b
        
                if res is True:
                    accepted += 1
                    RootLists[count].append(copy.deepcopy(Root))
                    
                    node_sums = 0
                    for k in np.arange(0,K):
                        node_sums += getNum(RootLists[k][-1])
                    nodeCounts.append(node_sums)
        
                    XX = np.zeros((n_train, K), dtype=self.tree_dtype)
                    for i in np.arange(K):
                        temp = allcal(RootLists[i][-1], train_data)
                        temp.shape = (temp.shape[0])
                        XX[:, i] = temp
                        
                    if np.iscomplexobj(XX):
                        XX = np.real(XX)
                        
                    constant = np.ones((n_train, 1))
                    XX = np.concatenate((constant, XX), axis=1)
                    scale = np.max(np.abs(XX), axis=0)
                    scale[scale == 0] = 1.0
                    XX = XX / scale
                    epsilon = np.eye(XX.shape[1]) * 1e-6
                    yy = np.array(train_y)
                    yy.shape = (yy.shape[0], 1)
                    Beta = np.linalg.inv(np.matmul(XX.transpose(), XX)+epsilon)
                    Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))
        
                    output = np.matmul(XX, Beta)
                    Beta = Beta / scale.reshape(-1, 1)
        
                    error = 0
                    for i in np.arange(0, n_train):
                        error += (output[i, 0] - train_y[i]) * (output[i, 0] - train_y[i])
                    rmse = np.sqrt(error / n_train)
                    errList.append(rmse)
                    
                    if self.disp:
                        print("accept", accepted, "th after", total, "proposals and update ", count, "th component")
                        print("sigma:", round(sigma, 5), "error:", round(rmse, 5))
            
                        display(genList(Root))
                        print("---------------")
                    totList.append(total)
                    total = 0
                    
                if len(errList) > 0 and errList[-1] < 1e-4:
                    switch_label = True
                    break
                my_index = min(10,len(errList))
                if len(errList)>100 and 1-np.min(errList[-my_index:])/np.mean(errList[-my_index:]) < 0.05:
                    switch_label = True
                    break
            if switch_label:
                break
            
        if self.disp:
            for i in np.arange(0,len(train_y)):
                print(output[i,0],train_y[i])
        
        toc = time.time()
        tictoc = toc-tic
        if self.disp:
            print("run time:{:.2f}s".format(tictoc))
            print("------")
            print("mean rmse of last 5 accepts:", np.mean(errList[-6:-1]))
            
        final_roots = [RootLists[c][-1] for c in np.arange(K)]
        return errList, final_roots, Beta

    # =============================================================================
    # # MCMC algorithm
    # K is the number of trees
    # MM is the number of iterations
    # alpha1, alpha2, beta are hyperparameters of priors
    # disp chooses whether to display intermediate results
        
    def fit(self, train_data, train_y):
        
        #WGL: moved these to fit and added underscore, 
        #since they are not user parameters
        self.roots_ = []
        self.betas_ = []
        self.train_err_ = []

        #WGL: train_data must be a dataframe
        if isinstance(train_data, np.ndarray):
            train_data = pd.DataFrame(train_data, dtype=self.tree_dtype)
        else:
            train_data = train_data.astype(self.tree_dtype)
        trainERRS = []
        #testERRS = []
        ROOTS = []
        BETAS = []
        MM = self.itrNum
       
        if self.disp: print('starting training...')
        n_feature = train_data.shape[1]
        n_train = train_data.shape[0]
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_one_iteration)(train_data, train_y, n_feature, n_train)
            for _ in range(MM)
        )
        
        # Sort results by final training error, descending, so the best model (lowest error) is at the end (-1)
        results.sort(key=lambda r: r[0][-1] if len(r[0]) > 0 else float('inf'), reverse=True)

        for errList, Roots, Beta in results:
            trainERRS.append(errList)
            ROOTS.append(Roots)
            BETAS.append(Beta)
            
        self.roots_ = ROOTS
        self.train_err_ = trainERRS
        self.betas_ = BETAS
            
        return
        

# =============================================================================
# # MCMC algorithm
# K is the number of trees
# MM is the number of iterations
# =============================================================================

def symreg(K,MM, train_data,test_data, train_y, test_y, disp=True):
    
    trainERRS = []
    testERRS = []
    ROOTS = []
    
    while len(trainERRS)<MM:
        
        
        n_feature = train_data.shape[1]
        n_train = train_data.shape[0]
        n_test = test_data.shape[0]
        
        alpha1 = 0.4
        alpha2 = 0.4
        beta = -1
        
        # Ops = ['inv', 'ln', 'neg', 'sin', 'cos', 'exp', '+', '*']
        # Op_weights = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        # Op_type = [1, 1, 1, 1, 1, 1, 2, 2]
        # n_op = len(Ops)
        
        Ops = ['inv', 'ln', 'neg', 'sin', 'cos', 'exp', 'square', 'cubic', '+', '*']
        Op_weights = [1.0/len(Ops)] * len(Ops)
        Op_type = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
        n_op = len(Ops)
        
        
        # List of tree samples
        RootLists = []
        for i in np.arange(K):
            RootLists.append([])
        
        SigaList = []  # List of sigma_a, for each component tree
        SigbList = []  # List of sigma_b, for each component tree
        
        sigma = invgamma.rvs(1)  # for output y
        
        val = 100
        
        # Initialization
        for count in np.arange(K):
            # create a new Root node
            Root = Node(0)
            sigma_a = invgamma.rvs(1)
            sigma_b = invgamma.rvs(1)
        
            # grow a tree from the Root node
            grow(Root, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
            # Tree = genList(Root)
        
            # put the root into list
            RootLists[count].append(copy.deepcopy(Root))
            SigaList.append(sigma_a)
            SigbList.append(sigma_b)
        
        # calculate beta
        # added a constant in the regression by fwl
        XX = np.zeros((n_train, K))
        for count in np.arange(K):
            temp = allcal(RootLists[count][-1], train_data)
            temp.shape = (temp.shape[0])
            XX[:, count] = temp
        constant = np.ones((n_train,1))  # added a constant
        XX = np.concatenate((constant, XX), axis=1)
        #scale = max(abs(np.max(XX)), abs(np.min(XX)))
        scale = np.max(np.abs(XX))
        XX = XX / scale
        epsilon = np.eye(XX.shape[1])*1e-6  # add to the matrix to prevent singular matrrix
        yy = np.array(train_y)
        yy.shape = (yy.shape[0], 1)
        Beta = np.linalg.inv(np.matmul(XX.transpose(), XX)+epsilon)
        Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))
        output = np.matmul(XX, Beta)
        Beta = Beta / scale  # rescale the beta, above we scale XX for calculation by fwl
        
        total = 0
        accepted = 0
        errList = []
        rootsList = []
        totList = []
        testList = []
        #testList2 = []
        dentList = []
        nodeCounts = []
        
        tic = time.time()
        
        while total < val:
            Roots = []  # list of current components
            # for count in np.arange(K):
            #     Roots.append(RootLists[count][-1])
            switch_label = False
            for count in np.arange(K):
                Roots = []  # list of current components
                for ccount in np.arange(K):
                    Roots.append(RootLists[ccount][-1])
                # pick the root to be changed
                sigma_a = SigaList[count]
                sigma_b = SigbList[count]
        
                oldRoot = copy.deepcopy(Roots[count])
                # the returned Root is a new copy
                [res, sigma, Root, sigma_a, sigma_b] = newProp(Roots, count, sigma, train_y, train_data, n_feature, Ops,
                                                               Op_weights, Op_type, beta, sigma_a, sigma_b)
                # print("res:",res)
                # display(genList(Root))
        
                total += 1
                # update sigma_a and sigma_b
                SigaList[count] = sigma_a
                SigbList[count] = sigma_b
        
                if res is True:
                    # flag = False
                    accepted += 1
                    # record newly accepted root
                    RootLists[count].append(copy.deepcopy(Root))
                    
                    node_sums = 0
                    for k in np.arange(0,K):
                        node_sums += getNum(RootLists[k][-1])
                    nodeCounts.append(node_sums)
        
                    XX = np.zeros((n_train, K))
                    for i in np.arange(K):
                        temp = allcal(RootLists[i][-1], train_data)
                        temp.shape = (temp.shape[0])
                        XX[:, i] = temp
                    constant = np.ones((n_train, 1))
                    XX = np.concatenate((constant, XX), axis=1)
                    scale = np.max(np.abs(XX))
                    XX = XX / scale
                    epsilon = np.eye(XX.shape[1]) * 1e-6  # add to the matrix to prevent singular matrrix
                    yy = np.array(train_y)
                    yy.shape = (yy.shape[0], 1)
                    Beta = np.linalg.inv(np.matmul(XX.transpose(), XX)+epsilon)
                    Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))
        
                    output = np.matmul(XX, Beta)
                    Beta = Beta / scale  # rescale the beta, above we scale XX for calculation
        
                    error = 0
                    for i in np.arange(0, n_train):
                        error += (output[i, 0] - train_y[i]) * (output[i, 0] - train_y[i])
                    rmse = np.sqrt(error / n_train)
                    errList.append(rmse)
                    
                    
                    # compute test error
                    XX = np.zeros((n_test, K))
                    for countt in np.arange(K):
                        temp = allcal(RootLists[countt][-1], test_data)
                        temp.shape = (temp.shape[0])
                        XX[:, countt] = temp
                    constant = np.ones((n_test, 1))
                    XX = np.concatenate((constant, XX), axis=1)
                    toutput = np.matmul(XX, Beta)
        
                    terror = 0
                    for i in np.arange(0, n_test):
                        terror += (toutput[i, 0] - test_y[i]) * (toutput[i, 0] - test_y[i])
                    trmse = np.sqrt(terror / n_test)
                    testList.append(trmse)
                    '''
                    # compute test2 error
                    XX = np.zeros((n_test, K))
                    for countt in np.arange(K):
                        temp = allcal(RootLists[countt][-1], test2_data)
                        temp.shape = (temp.shape[0])
                        XX[:, countt] = temp
                    constant = np.ones((n_test, 1))
                    XX = np.concatenate((constant, XX), axis=1)
                    toutput2 = np.matmul(XX, Beta)
        
                    terror2 = 0
                    for i in np.arange(0, n_test):
                        terror2 += (toutput2[i, 0] - test2_y[i]) * (toutput2[i, 0] - test2_y[i])
                    trmse2 = np.sqrt(terror2 / n_test)
                    testList2.append(trmse2)
                    '''
                    
                    if disp:

                        print("accept", accepted, "th after", total, "proposals and update ", count, "th component")
                        print("sigma:", round(sigma, 5), "error:", round(rmse, 5))  # ,"log.likelihood:",round(llh,5))
                        # print("denoised rmse:",round(dtrmse,5))
            
                        display(genList(Root))
                        print("---------------")
                    totList.append(total)
                    total = 0
                    
                    
        
                # @fwl added condition to control running time
                my_index = min(10,len(errList))
                if len(errList)>100 and 1-np.min(errList[-my_index:])/np.mean(errList[-my_index:]) < 0.05:
                    # converged
                    switch_label = True
                    break
                    # Roots[count] = oldRoot
            if switch_label:
                break
            
        for i in np.arange(0,len(train_y)):
            print(output[i,0],train_y[i])
        
        toc = time.time() # cauculate running time
        tictoc = toc-tic
        if disp:
            print("run time:{:.2f}s".format(tictoc))
        
            print("------")
            print("mean rmse of last 5 accepts:", np.mean(errList[-6:-1]))
            print("mean rmse of last 5 tests:", np.mean(testList[-6:-1]))
        
        trainERRS.append(errList)
        testERRS.append(testList)
        ROOTS.append(Roots)
        
    return(ROOTS)
