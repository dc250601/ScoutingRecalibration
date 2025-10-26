import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import warnings 

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from qkeras import QBatchNormalization
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

import hls4ml

#---------------------------------------------------------------
#Settings
#---------------------------------------------------------------


#-----------------------------------------------------------------
#Tensorflow GPU Settings chnage them as per your need
use_GPU = True
pd.set_option("display.max_columns", None)
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
#-----------------------------------------------------------------
    

#-----------------------------------------------------------------
#Matplotlib fontsizes
plt.rc('font',   size      = 16)    # controls default text sizes
plt.rc('axes',   titlesize = 18)    # fontsize of the axes title
plt.rc('axes',   labelsize = 16)    # fontsize of the x and y labels
plt.rc('xtick',  labelsize = 12)    # fontsize of the tick labels
plt.rc('ytick',  labelsize = 12)    # fontsize of the tick labels
plt.rc('legend', fontsize  = 12)    # legend fontsize
plt.rc('figure', titlesize = 18)    # fontsize of the figure title
#-----------------------------------------------------------------


#-----------------------------------------------------------------


def add_integer_values_to_df(df_base):
    """
    Function:
    convert float phi,eta,pt value to integer values
    https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/master/doc/scales_inputs_2_ugt/pdf/scales_inputs_2_ugt.pdf
    """

    new_phi_e = np.array([x if x > 0 else x + 2*math.pi for x in df_base['phiVtxL1']])
    
    df_base['int_phi_e'] = (new_phi_e / (2*math.pi / 576)).astype(int)
    df_base['int_eta_e'] = (df_base['etaVtxL1'] / 0.010875).astype(int)
    df_base['int_pt_r']  = (df_base['pTL1'] / 0.5).astype(int)
    
    return df_base


def calculate_delta_phi(phi_approx, phiRecorue):
    """
    Function:
    wrap phase for phi values and then compute difference with reco value
    """
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x for x in phi_approx - phiRecorue])


def calculate_phi(phi_approx, delta_phi):
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x for x in phi_approx - delta_phi])


def load_data(file_paths, sample_fraction = 1):
    df_list = []
    for files in file_paths:
        df_list.append(pd.read_csv(files))
    df_base = pd.concat(df_list, axis = 0)
    df_base   = df_base.sample(frac=sample_fraction)
    return df_base

def load_data(file_paths, sample_fraction = 1):
    df_list = []
    for files in file_paths:
        df_list.append(pd.read_csv(files))
    df_base = pd.concat(df_list, axis = 0)
    df_base   = df_base.sample(frac=sample_fraction)
    return df_base

def apply_cut(df_frame, lower_limit, upper_limit):
    warnings.warn("The limits are expected to be the exact float values and not the converted integer counter parts")
    df_frame = df_frame[df_frame.hwPtL1Mu1<2*(upper_limit)]
    df_frame = df_frame[df_frame.hwPtL1Mu1>2*(lower_limit)]
    df_frame = df_frame[df_frame.hwPtL1Mu2<2*(upper_limit)]
    df_frame = df_frame[df_frame.hwPtL1Mu2>2*(lower_limit)]
    return df_frame


def plot_data(df_base):
    """
    Function to plot the input data
    """
    rebin=1
    plt.figure(figsize=(30,13))

    plt.subplot(2,3,1)
    x_lim = [0, 575+1]
    nbins = x_lim[1] - x_lim[0]
    plt.hist(df_base.hwPhiL1Mu1, bins=nbins//rebin, alpha=1, histtype='step', color='C0', fill=False, range=x_lim)
    plt.xlim(*x_lim)
    plt.xlabel(" $\\varphi$")
    plt.ylabel("Counts")
    # plt.title(' $\phi$ distribution')

    plt.subplot(2,3,2)
    x_lim = [-256, 255+1]
    nbins = x_lim[1] - x_lim[0]
    plt.hist(df_base.hwEtaL1Mu1, bins=nbins//rebin, alpha=1, histtype='step', color='C0', fill=False, range=x_lim)
    plt.xlim(*x_lim)
    plt.xlabel("$\eta$")
    plt.ylabel("Counts")
    # plt.title(' $\eta$ distribution')

    plt.subplot(2,3,3)
    x_lim = [0, 128]
    nbins = x_lim[1] - x_lim[0]
    plt.hist(df_base.hwPtL1Mu1, bins=nbins//rebin, alpha=1, histtype='step', color='C0', fill=False, range=x_lim)
    plt.xlabel("$p_T$")
    plt.ylabel("Counts")
    plt.yscale("log")
    # plt.title(' $p_T$ distribution')

    plt.subplot(2,3,4)
    x_lim = [0, 575+1]
    nbins = x_lim[1] - x_lim[0]
    plt.hist(df_base.deltaPhi1_int, bins=nbins//rebin, alpha=1, histtype='step', color='C0', fill=False, range=x_lim)
    plt.xlabel("$\Delta \\varphi$")
    plt.ylabel("Counts")
    # plt.title('$\Delta \phi$ distribution')

    plt.subplot(2,3,5)
    x_lim = [-50, 50]
    nbins = x_lim[1] - x_lim[0]
    plt.hist(df_base.deltaEta1_int, bins=nbins//rebin, alpha=1, histtype='step', color='C0', fill=False, range=x_lim)
    plt.xlabel("$ \Delta \eta$")
    plt.ylabel("Counts")
    # plt.title(' $ \Delta \eta$ distribution')

    plt.subplot(2,3,6)
    x_lim = [-50, 50]
    nbins = x_lim[1] - x_lim[0]
    plt.hist(df_base.deltaPt1_int, bins=nbins//rebin, alpha=1, histtype='step', color='C0', fill=False, range=x_lim)
    plt.xlabel("$\Delta p_T$")
    plt.ylabel("Counts")
    plt.yscale("log")
    # plt.title('$\Delta p_T$ distribution')
    plt.show() 
    
def preprocess_data(df, phi_div  = 64, eta_div  = 64, pt_div   = 64, qual_div = 64):
    Length  = len(df)
    print("Length = ", Length)
    x = np.array(df[['hwPhiL1Mu1', 'hwEtaL1Mu1', 'hwPtL1Mu1',
                             'hwSignL1Mu1','hwQualityL1Mu1',
                             'hwPhiL1Mu2', 'hwEtaL1Mu2', 'hwPtL1Mu2',
                             'hwSignL1Mu2', 'hwQualityL1Mu2']])/ phi_div


    y = np.array((df[["label"]]))
    
    return x,y