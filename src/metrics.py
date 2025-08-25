import pandas as pd
import numpy as np 
import os 
import sys
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, auc, confusion_matrix
from torch.utils.data import Dataset, DataLoader


def compute(ytrue, ypred, threshold):
    acc = compute_acc(ytrue, ypred, threshold)
    auroc = compute_auroc(ytrue, ypred)
    CM = compute_CM(ytrue, ypred, threshold)
    aupr = compute_aupr(ytrue, ypred, threshold)
    return acc, auroc, aupr, CM

def compute_acc(ytrue, ypred, threshold):
    ypred = (ypred >threshold).float()
    matches = (ypred == ytrue)
    acc = torch.sum(matches) / len(ypred)
    return acc

def compute_auroc(ytrue, ypred):
    ## rocauc requires raw probs
    return roc_auc_score(ytrue, ypred)

def compute_CM(ytrue, ypred, threshold):
    ypred = (ypred >threshold).float()
    ## tn, fp, fn, tp 
    cm = confusion_matrix(ytrue, ypred,labels=[0,1])
    return cm

def compute_aupr(ytrue, ypred, threshold):
    ypred = (ypred >threshold).float()
    return average_precision_score(ytrue, ypred)
