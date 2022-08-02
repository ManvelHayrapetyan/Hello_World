# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:56:39 2022

@author: Manvel Hayrapetyan
"""


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,\
                                            roc_auc_score, confusion_matrix, mean_absolute_error               

                        
class Metrics:
    '''
    Take a true and predicted data(one_hot or normal)
    Retrun common metrics results
    '''  
    def __init__(self, y_true, y_pred):
        
        self.y_true = y_true
        self.y_pred = y_pred
        
        # remove one hot
        if y_true.ndim == 1:
            self.remove_onehot_true = False
        elif y_true.ndim == 2:
            self.remove_onehot_true = True
        else:
            raise TypeError('y_true sxal chaperia')
            
        if y_pred.ndim == 1:
            self.remove_onehot_pred = False
        elif y_pred.ndim == 2:
            self.remove_onehot_pred = True
        else:
            raise TypeError('y_pred sxal chaperia')
        
        self.y_true_removed = y_true.argmax(axis=1) if self.remove_onehot_true else y_true
        self.y_pred_removed = y_pred.argmax(axis=1) if self.remove_onehot_pred else y_pred
        
        
    def acc(self):
        return accuracy_score(self.y_true_removed, self.y_pred_removed)
    
    def f1(self, average='macro'):
        return f1_score(self.y_true_removed, self.y_pred_removed, average=average, zero_division=0)
    
    
    def recall(self, average='macro'):
        return recall_score(self.y_true_removed, self.y_pred_removed, average=average, zero_division=0)
    
    
    def precision(self, average='macro'):
        return precision_score(self.y_true_removed, self.y_pred_removed, average=average, zero_division=0)
    
    
    def auc(self):
        if self.remove_onehot_true != True or self.remove_onehot_pred != True:
            raise ValueError('we need one hot vectors')
        else:
            return roc_auc_score(self.y_true, self.y_pred, average='macro', multi_class='ovo')
        
        
    def loss(self, loss):
        return float(loss(self.y_true, self.y_pred))
    
    
    def mae(self):
        return mean_absolute_error(self.y_true_removed, self.y_pred_removed)
        
      
    def add_1_inaccuracy(self): 
        '''
        WORK WITH ERRORS, REPAIR IT
        we will assume that the wrong answer,
        which was only 1 class distance wrong,
        will be correct. And compute Acc and F1 score
        '''
        y_pred_1 = []
        
        for i in range(self.y_true_removed.shape[0]):
            if abs(self.y_true_removed[i] - self.y_pred_removed[i]) <= 1:
                y_pred_1.append(self.y_true_removed[i])
            else:
                y_pred_1.append(self.y_pred_removed[i])
                
        return [accuracy_score(self.y_true_removed, y_pred_1),
                f1_score(self.y_true_removed, y_pred_1, average='macro', zero_division=0)]
    
    
    def confusion(self):
        plt.figure(figsize = (10,6))
        cm = confusion_matrix(self.y_true_removed, self.y_pred_removed)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt="d");
        return cm
        

def results(y_true, y_pred, *args, **kwargs):
    '''
    Compute list of metrics of given true and pred labels(one_hot or normal)
    when you use loss, don't forget to add loss function
    all metrics available
    'acc',
    'add_1_inaccuracy', 
    'auc',
    'confusion',
    'f1',
    'loss',
    'mae',
    'precision',
    'recall'
    
    e.g. results(aa,bb, 'acc', 'loss', 'mae', loss=loss())
    '''
    result = Metrics(y_true, y_pred)
    for function in args:
        if function == 'loss':
            print(function,'\t', result.loss(kwargs['loss']))
        elif function == 'add_1_inaccuracy':
            print('acc±1\t',result.add_1_inaccuracy()[0])
            print('f1±1\t',result.add_1_inaccuracy()[1])
        else:
            print(function,'\t',getattr(result, function)())


def results_for_log(y_true, y_pred, loss, add_label='', one_hot=True):
    '''
    results in dict format, use to save in wandb
    '''
    result = Metrics(y_true, y_pred)
    functions = ['acc', 'f1', 'precision', 'recall', 'mae', 'auc']
    functions = functions if one_hot else functions[:-1] # hanuma AUC vor error chta
    dct = {add_label + '_' + function: getattr(result, function)() for function in functions}
    dct.update(dict(zip([add_label + '_' + 'acc±1', add_label + '_' + 'f1±1'], result.add_1_inaccuracy())))
    dct.update({add_label + '_loss':result.loss(loss)})
    
    return dct


def results_x2(y_true_1, y_pred_1, y_true_2, y_pred_2, name_1 ='', name_2 =''):
    '''
    see the results of 2 datas(models,...)
    e.g.(train,valid), (model1_valid, model2_valid)
    '''
    
    result1 = Metrics(y_true_1, y_pred_1)
    result2 = Metrics(y_true_2, y_pred_2)
    
    print(name_1,'  acc ', round(result1.acc(),4), ' f1 ', round(result1.f1(),4))
    print(name_2,'  acc ', round(result2.acc(),4), ' f1 ', round(result2.f1(),4))
    
    #confusion
    fig, axs= plt.subplots(1,2, figsize=(24, 6))
    axs[0].set_title(name_1, size = 24)
    cm = confusion_matrix(result1.y_true_removed, result1.y_pred_removed)
    sns.heatmap(cm, annot=True, ax = axs[0], cmap='Blues', fmt="d")
    axs[1].set_title(name_2, size = 24)
    cm = confusion_matrix(result2.y_true_removed, result2.y_pred_removed)
    sns.heatmap(cm, annot=True, ax = axs[1], cmap='Blues', fmt="d");