import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from xml.dom import minidom
import math
import os
from io import BytesIO
from vagen.env.svg.dino import DINOScoreCalculator


def calculate_structural_accuracy(gt_im, gen_im):
    "range from 0 - 1"
    gt_gray = np.array(gt_im.convert('L'))
    gen_gray = np.array(gen_im.convert('L'))
    
    gt_edges = cv2.Canny(gt_gray, 100, 200)
    gen_edges = cv2.Canny(gen_gray, 100, 200)
    
    intersection = np.logical_and(gt_edges, gen_edges).sum()
    union = np.logical_or(gt_edges, gen_edges).sum()
    
    return intersection / union if union > 0 else 0


def calculate_color_fidelity(self, gt_im, gen_im):
    "range from 0 - 1"
    gt_lab = cv2.cvtColor(np.array(gt_im), cv2.COLOR_RGB2LAB)
    gen_lab = cv2.cvtColor(np.array(gen_im), cv2.COLOR_RGB2LAB)
    
    mse = np.mean((gt_lab - gen_lab) ** 2)
    sim = np.exp(-mse / 1000)
    return sim


def calculate_code_efficiency(self, gt_code, gen_code):
    if not gen_code:
      return 0
    gt_len = len(gt_code)
    gen_len = len(gen_code)

    if gen_len == gt_len:
        return 0.8

    if gen_len < gt_len:
        ratio = gen_len / gt_len 
        score = 0.8 + 0.2 * (1 - ratio)
        return min(score, 1.0)

    ratio = gt_len / gen_len 
    score = 0.8 * ratio
    return max(score, 0.0)
  

#@TODO make it into class?
def calculate_total_score(reward_model, gt_im, gen_im, gt_code, gen_code, dino_only=False):
    """
    calculate all metrics on average
    
    Args:
        reward_model: DINO Model
        gt_im: gt image
        gen_im: generated image
        gen_svg: generated code
        dino_only: whether only use dino as score
        
    Returns:
        dict: include all scores
    """
    dino_score = reward_model.calculate_DINOv2_similarity_score(gt_im=gt_im, gen_im=gen_im)
    
    if dino_only:
        return {
            'dino_score': dino_score,
            'total_score': dino_score
        }
    
    structural_score = calculate_structural_accuracy(gt_im, gen_im)
    color_score = calculate_color_fidelity(gt_im, gen_im)
    code_score = calculate_code_efficiency(gt_code, gen_code)
    
    weights = {
        'dino_score': 5.0,
        'structural_accuracy': 3.0,
        'color_fidelity': 2.0,
        'code_efficiency': 2.0
    }
    
    scores = {
        'dino_score': dino_score,
        'structural_accuracy': structural_score,
        'color_fidelity': color_score,
        'code_efficiency': code_score
    }
    
    weighted_sum = sum(scores[k] * weights[k] for k in weights)
        
    scores['total_score'] = max(0.0, weighted_sum)
    
    return scores