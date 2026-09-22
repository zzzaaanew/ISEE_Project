"""Validation-only highest-threshold recall operating points."""
import math
import numpy as np


def select_threshold(scores,labels,target=.99,*,split='validation'):
    if split!='validation':raise ValueError('Threshold selection accepts validation only')
    scores=np.asarray(scores,dtype=float);labels=np.asarray(labels)
    if scores.ndim!=1 or scores.shape!=labels.shape or not 0<target<=1:raise ValueError('Invalid threshold inputs')
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any() or not np.isin(labels,[0,1]).all():raise ValueError('Invalid score/label')
    positives=np.sort(scores[labels==1])
    if not len(positives):return {'status':'infeasible','reason':'no_validation_positives','target_recall':target}
    threshold=float(positives[len(positives)-math.ceil(target*len(positives))])
    return {'status':'feasible','threshold':threshold,'target_recall':target,'recall':float(np.mean(positives>=threshold)),'selection_split':'validation_only'}
