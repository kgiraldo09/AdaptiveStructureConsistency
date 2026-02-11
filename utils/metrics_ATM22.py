'''
Program :   The evaluation functions for the ATM 22 Challenge, including the TD / BD / DSC / Precision
Author  :   Minghui Zhang, Institute of Medical Robotics, Shanghai Jiao Tong University.
File    :   evaluation_atm22.py
Date    :   2022/02/02 16:19
Version :   V1.0
'''

import numpy as np


def branch_detected_calculation(pred, label_parsing, label_skeleton, thresh=0.8):
    label_branch = label_skeleton * label_parsing
    label_branch_flat = label_branch.flatten()
    label_branch_bincount = np.bincount(label_branch_flat)[1:]
    total_branch_num = label_branch_bincount.shape[0]
    pred_branch = label_branch * pred
    pred_branch_flat = pred_branch.flatten()
    pred_branch_bincount = np.bincount(pred_branch_flat)[1:]
    if total_branch_num != pred_branch_bincount.shape[0]:
        lack_num = total_branch_num - pred_branch_bincount.shape[0]
        pred_branch_bincount = np.concatenate((pred_branch_bincount, np.zeros(lack_num)))
    branch_ratio_array = pred_branch_bincount / label_branch_bincount
    branch_ratio_array = np.where(branch_ratio_array >= thresh, 1, 0)
    detected_branch_num = np.count_nonzero(branch_ratio_array)
    detected_branch_ratio = round((detected_branch_num * 100) / total_branch_num, 2)
    return total_branch_num, detected_branch_num, detected_branch_ratio


def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = ((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100
    return dice_coefficient_score


def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
    pred = pred.flatten()
    label_skeleton = label_skeleton.flatten()
    tree_length = (np.sum(pred * label_skeleton) + smooth) / (np.sum(label_skeleton) + smooth) * 100
    return tree_length


def false_positive_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fp = np.sum(pred - pred * label) + smooth
    fpr = fp * 100 / (np.sum((1.0 - label)) + smooth)
    return fpr


def false_negative_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fn = np.sum(label - pred * label) + smooth
    fnr = fn * 100 / (np.sum(label) + smooth)
    return fnr


def sensitivity_calculation(pred, label):
    sensitivity = TP(pred, label) / (TP(pred, label) + FN(pred, label))
    return sensitivity

def TP(pred, label):
    return np.sum(pred*label)

def FN(pred, label):
    return np.sum((1 - pred)*label)

def TN(pred, label):
    return np.sum((1 - pred)*(1 - label))

def FP(pred, label):
    return np.sum(pred *(1 - label))

def specificity_calculation(pred, label):

    specificity = 100*TN(pred, label) / (TN(pred, label) + FP(pred, label))
    return specificity


def precision_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    tp = np.sum(pred * label) + smooth
    precision = tp * 100 / (np.sum(pred) + smooth)
    return precision


from skimage.morphology import skeletonize
#, skeletonize_3d
import numpy as np

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 100*2*tprec*tsens/(tprec+tsens)
