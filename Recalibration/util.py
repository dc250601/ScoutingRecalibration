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

def apply_cut(df_frame, lower_limit, upper_limit):
    warnings.warn("The limits are expected to be the exact float values and not the converted integer counter parts")
    df_frame = df_frame[df_frame.hwPtL1Mu1<2*(upper_limit)]
    df_frame = df_frame[df_frame.hwPtL1Mu1>2*(lower_limit)]
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
    

def preprocess_data(df_base, train_perc = 0.95, phi_div  = 64, eta_div  = 64, pt_div   = 64, qual_div = 64):
    train_len  = int(train_perc * len(df_base))
    print("train_len = ", train_len)
    df_train = df_base[:train_len]
    df_test  = df_base[train_len:]

    print("Train df shape:", df_train.shape)
    print("Test  df shape:", df_test.shape)

    # "normalize"
    # on hardware division by 256 is easy to implement
    x_train = np.concatenate(
        (
            np.array(df_train[["hwPhiL1Mu1"]]) / phi_div,
            np.array(df_train[["hwEtaL1Mu1"]]) / eta_div,
            np.array(df_train[["hwPtL1Mu1"]])  / pt_div ,
            np.array(df_train[["hwSignL1Mu1"]]),
            np.array(df_train[["hwQualityL1Mu1"]]) / qual_div
        ),
        axis=1
    )
    y_train = np.concatenate(
        (
            np.array((df_train[["deltaPhi1_int"]] + 288) % 576 - 288) / phi_div,
            # np.array(df_train[["deltaPhi1_int"]]) / phi_div,
            np.array(df_train[["deltaEta1_int"]]) / eta_div,
            np.array(df_train[["deltaPt1_int"]])  / pt_div 
        ),
        axis=1
    )

    x_test = np.concatenate(
        (
            np.array(df_test[["hwPhiL1Mu1"]]) / phi_div,
            np.array(df_test[["hwEtaL1Mu1"]]) / eta_div,
            np.array(df_test[["hwPtL1Mu1"]])  / pt_div ,
            np.array(df_test[["hwSignL1Mu1"]]),
            np.array(df_test[["hwQualityL1Mu1"]]) / qual_div
        ),
        axis=1
    )
    y_test = np.concatenate(
        (
            np.array((df_test[["deltaPhi1_int"]] + 288) % 576 - 288) / phi_div,
            # np.array(df_test[["deltaPhi1_int"]]) / phi_div,
            np.array(df_test[["deltaEta1_int"]]) / eta_div,
            np.array(df_test[["deltaPt1_int"]])  / pt_div 
        ),
        axis=1
    )

    print("X train shape:", x_train.shape)
    print("Y train shape:", y_train.shape)
    print("X test  shape:", x_test.shape)
    print("Y test  shape:", y_test.shape)
    
    return ((x_train,y_train),(x_test,y_test), df_train, df_test)


def inv_phi_range(arr_phi):
    return np.array([(x+576) if x < 0 else x for x in arr_phi ])

def evaluate_corrector(model,
                   x_test,
                   y_test,
                   df_train,
                   df_test,
                   phi_div = 64,
                   eta_div = 64,
                   pt_div = 64,
                   qual_div = 64,
                   batch_size = 2048):
    
    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred = np.squeeze(y_pred)
    delta_pt_pred =  y_pred * pt_div * 0.5


    pt_pred= np.array(df_test['PtL1Mu1'][:] - delta_pt_pred[:])

    delta_pt_p = np.array(pt_pred - df_test['ptRecoMu1'][:])


    plt.figure(figsize=(9,6))
    plt.hist(delta_pt_p/df_test['ptRecoMu1'],            bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN")
    plt.hist((df_test['deltaPt1']/df_test['ptRecoMu1']), bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT")
    plt.xlabel("$\Delta p_{T}(reco,pred)/p_{T}^{reco}$")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    
    pt = np.array(delta_pt_p/df_test['ptRecoMu1'])
    
    print("FWHM of delta_pt for the Neural Network Qkeras", find_fwhm(pt))


def evaluate_keras(model,
                   x_test,
                   y_test,
                   df_train,
                   df_test,
                   phi_div = 64,
                   eta_div = 64,
                   pt_div = 64,
                   qual_div = 64,
                   batch_size = 2048):
    
    y_pred = model.predict(x_test, batch_size=batch_size)
    y_pred = np.squeeze(y_pred)
    y_pred[:,0] = inv_phi_range(np.array(y_pred[:,0])*phi_div)
    delta_phi_pred = y_pred[:,0] * (2*math.pi / 576) 
    delta_phi_pred = np.array([x-2*np.pi if x > np.pi else x for x in delta_phi_pred])
    delta_eta_pred = y_pred[:,1] * eta_div * 0.010875
    delta_pt_pred =  y_pred[:,2] * pt_div * 0.5

    phi_pred = calculate_phi(df_test['PhiL1Mu1'][:], delta_phi_pred[:])
    eta_pred = np.array(df_test['EtaL1Mu1'][:] - delta_eta_pred[:])
    pt_pred= np.array(df_test['PtL1Mu1'][:] - delta_pt_pred[:])

    delta_phi_p = calculate_phi(phi_pred, df_test['phiVtxRecoMu1'][:])
    delta_eta_p = np.array(eta_pred - df_test['etaVtxRecoMu1'][:])
    delta_pt_p = np.array(pt_pred - df_test['ptRecoMu1'][:])

    delta_phi_ext = calculate_phi(df_test['phiAtVtx1'], df_test['phiVtxRecoMu1'])
    delta_eta_ext = df_test['etaAtVtx1'] - df_test['etaVtxRecoMu1']
    
    plt.figure(figsize=(9,6))
    plt.hist(delta_phi_p,            bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN")
    plt.hist(delta_phi_ext,          bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT EXT")
    plt.hist((df_test['deltaPhi1']), bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT")
    plt.xlabel("$\Delta\\varphi(reco,pred)$")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,6))
    plt.hist(delta_eta_p,            bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN")
    plt.hist(delta_eta_ext,          bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT EXT")
    plt.hist((df_test['deltaEta1']), bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT")
    plt.xlabel("$\Delta\eta(reco,pred)$")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,6))
    plt.hist(delta_pt_p/df_test['ptRecoMu1'],            bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN")
    plt.hist((df_test['deltaPt1']/df_test['ptRecoMu1']), bins=100, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT")
    plt.xlabel("$\Delta p_{T}(reco,pred)/p_{T}^{reco}$")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    
    phi  = delta_phi_p
    eta = delta_eta_p
    pt = np.array(delta_pt_p/df_test['ptRecoMu1'])
    
    print("FWHM of delta_phi for the Neural Network Qkeras",find_fwhm(phi))
    print("FWHM of delta_eta for the Neural Network Qkeras", find_fwhm(eta))
    print("FWHM of delta_pt for the Neural Network Qkeras", find_fwhm(pt))
    
    
def evaluate_hls(hls_model,
                 keras_model,
                 x_test,
                 y_test,
                 df_train,
                 df_test,
                 phi_div = 64,
                 eta_div = 64,
                 pt_div = 64,
                 qual_div = 64,
                 batch_size = 2048
                ):
    
    y_pred = keras_model.predict(x_test, batch_size=batch_size)
    y_pred = np.squeeze(y_pred)
    y_pred[:,0] = inv_phi_range(np.array(y_pred[:,0])*phi_div)
    delta_phi_pred = y_pred[:,0] * (2*math.pi / 576) 
    delta_phi_pred = np.array([x-2*np.pi if x > np.pi else x for x in delta_phi_pred])
    delta_eta_pred = y_pred[:,1] * eta_div * 0.010875
    delta_pt_pred =  y_pred[:,2] * pt_div * 0.5

    phi_pred = calculate_phi(df_test['PhiL1Mu1'][:], delta_phi_pred[:])
    eta_pred = np.array(df_test['EtaL1Mu1'][:] - delta_eta_pred[:])
    pt_pred= np.array(df_test['PtL1Mu1'][:] - delta_pt_pred[:])

    delta_phi_p = calculate_phi(phi_pred, df_test['phiVtxRecoMu1'][:])
    delta_eta_p = np.array(eta_pred - df_test['etaVtxRecoMu1'][:])
    delta_pt_p = np.array(pt_pred - df_test['ptRecoMu1'][:])    
    
    y_hls = hls_model.predict(x_test)
    delta_phi_hls = inv_phi_range(np.array(y_hls[:,0])*phi_div)
    delta_phi_hls = delta_phi_hls * (2*math.pi / 576) 
    delta_phi_hls = np.array([x-2*np.pi if x > np.pi else x for x in delta_phi_hls])
    delta_eta_hls = y_hls[:,1] * eta_div * 0.010875
    delta_pt_hls =  y_hls[:,2] * pt_div * 0.5
    
    
    phi_hls = calculate_phi(df_test['PhiL1Mu1'][:], delta_phi_hls[:])
    eta_hls = np.array(df_test['EtaL1Mu1'][:] - delta_eta_hls[:])
    pt_hls= np.array(df_test['PtL1Mu1'][:] - delta_pt_hls[:])
    
    delta_phi_p_hls = calculate_phi(phi_hls, df_test['phiVtxRecoMu1'][:])
    delta_eta_p_hls = np.array(eta_hls - df_test['etaVtxRecoMu1'][:])
    delta_pt_p_hls = np.array(pt_hls - df_test['ptRecoMu1'][:])

    delta_phi_ext = calculate_phi(df_test['phiAtVtx1'], df_test['phiVtxRecoMu1'])
    delta_eta_ext = df_test['etaAtVtx1'] - df_test['etaVtxRecoMu1']
    
    nbins_phi = 200
    nbins_eta = 200
    nbins_pt  = 200

    plt.figure(figsize=(9,6))
    plt.hist(delta_phi_p,            bins=nbins_phi, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN")
    plt.hist(delta_phi_p_hls,        bins=nbins_phi, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN HLS4ML")
    plt.hist(delta_phi_ext,          bins=nbins_phi, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT EXT")
    plt.hist((df_test['deltaPhi1']), bins=nbins_phi, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT")
    plt.xlabel("$\Delta\\varphi(\mathrm{reco},\mathrm{pred})$")
    plt.ylabel("Density")
    plt.legend()
    
    plt.show()
    plt.figure(figsize=(9,6))
    plt.hist(delta_eta_p,            bins=nbins_eta, range=[-0.25,0.25], alpha=1.0, histtype="step", density=True, label="NN")
    plt.hist(delta_eta_p_hls,        bins=nbins_eta, range=[-0.25,0.25], alpha=1.0, histtype="step", density=True, label="NN HLS4ML")
    plt.hist(delta_eta_ext,          bins=nbins_eta, range=[-0.25,0.25], alpha=1.0, histtype="step", density=True, label="uGMT EXT")
    plt.hist((df_test['deltaEta1']), bins=nbins_eta, range=[-0.25,0.25], alpha=1.0, histtype="step", density=True, label="uGMT")
    plt.xlabel("$\Delta\eta(\mathrm{reco},\mathrm{pred})$")
    plt.ylabel("Density")
    plt.legend()
    # plt.savefig(os.path.join(fig_path,"eta.png"))
    plt.show()
    # plt.close()
    plt.figure(figsize=(9,6))
    plt.hist(delta_pt_p/df_test['ptRecoMu1'],            bins=nbins_pt, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN")
    plt.hist(delta_pt_p_hls/df_test['ptRecoMu1'],        bins=nbins_pt, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN HLS4ML")
    plt.hist((df_test['deltaPt1']/df_test['ptRecoMu1']), bins=nbins_pt, range=[-1,1], alpha=1.0, histtype="step", density=True, label="uGMT")
    plt.xlabel("$\Delta p_{T}(\mathrm{reco},\mathrm{pred})/p_{T}^{\mathrm{reco}}$")
    plt.ylabel("Density")
    plt.legend()
    # plt.savefig(os.path.join(fig_path,"pt.png"))
    plt.show()
    # plt.close()
    
    phi  = delta_phi_p
    eta = delta_eta_p
    pt = np.array(delta_pt_p/df_test['ptRecoMu1'])

    phi_h  = delta_phi_p_hls
    eta_h = delta_eta_p_hls
    pt_h = np.array(delta_pt_p_hls/df_test['ptRecoMu1'])

    
    print("FWHM of delta_phi for the Neural Network Qkeras",find_fwhm(phi))
    print("FWHM of delta_eta for the Neural Network Qkeras", find_fwhm(eta))
    print("FWHM of delta_pt for the Neural Network Qkeras", find_fwhm(pt))
    print()
    print("FWHM of delta_phi for the Neural Network HLS4ML",find_fwhm(phi_h))
    print("FWHM of delta_eta for the Neural Network HLS4ML", find_fwhm(eta_h))
    print("FWHM of delta_pt for the Neural Network HLS4ML", find_fwhm(pt_h))

    
    
def find_fwhm(data):
    stats = plt.hist(data, bins=2000, range=[-1,1], alpha=1.0, histtype="step", density=True, label="NN")
    plt.close()
    y,x,_ = stats
    x_ = []
    for i in range(len(x) - 1):
        x_.append((x[i]+x[i+1])/2)
    x = x_


    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    parameters,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    x_fit = np.arange(min(x),max(x),0.001)
    y_fit = Gauss(x_fit,parameters[0], parameters[1], parameters[2])
    max_height = y_fit.max()
    x_l, x_r = x_fit[y_fit>(max_height/2)][0], x_fit[y_fit>(max_height/2)][-1]
    fwhm = x_r - x_l
    return fwhm


def lin_interp(x, y, i, half):
    return (x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i])),i)    

#-----------------------------------------------------------------

#-----------------------------------------------------------------
#Callbacks
#-----------------------------------------------------------------
class FWHM(tf.keras.callbacks.Callback):
    def __init__(self,
                 test_x,
                 test_y,
                 df,
                 use_wandb = False,
                 verbose = True,
                 phi_div = 64,
                 eta_div = 64,
                 pt_div = 64,
                 qual_div = 64):
        
        super().__init__()
        self.x_test = test_x
        self.test_y = test_y
        self.df_test = df
        self.use_wandb = use_wandb
        self.verbose = verbose
        self.phi_div = phi_div
        self.eta_div = eta_div
        self.pt_div = pt_div
        self.qual_div = qual_div
    

    def on_epoch_end(self, epoch, logs=None):
        y_pred =  self.model.predict(self.x_test)
        y_pred[:,0] = inv_phi_range(np.array(y_pred[:,0])*self.phi_div)
        delta_phi_pred = y_pred[:,0] * (2*math.pi / 576) 
        delta_phi_pred = np.array([x-2*np.pi if x > np.pi else x for x in delta_phi_pred])
        delta_eta_pred = y_pred[:,1] * self.eta_div * 0.010875
        delta_pt_pred =  y_pred[:,2] * self.pt_div * 0.5

        phi_pred = calculate_phi(self.df_test['PhiL1Mu1'][:], delta_phi_pred[:])
        eta_pred = np.array(self.df_test['EtaL1Mu1'][:] - delta_eta_pred[:])
        pt_pred= np.array(self.df_test['PtL1Mu1'][:] - delta_pt_pred[:])

        delta_phi_p = calculate_phi(phi_pred, self.df_test['phiVtxRecoMu1'][:])
        delta_eta_p = np.array(eta_pred - self.df_test['etaVtxRecoMu1'][:])
        delta_pt_p = np.array(pt_pred - self.df_test['ptRecoMu1'][:])

        delta_phi_ext = calculate_phi(self.df_test['phiAtVtx1'], self.df_test['phiVtxRecoMu1'])
        delta_eta_ext = self.df_test['etaAtVtx1'] - self.df_test['etaVtxRecoMu1']

        fwhm_phi_NN = find_fwhm(delta_phi_p)
        fwhm_phi_uGMT_Ext = find_fwhm(delta_phi_ext)
        fwhm_phi_uGMT = find_fwhm(np.array((self.df_test['deltaPhi1'])))

        fwhm_eta_NN = find_fwhm(delta_eta_p)
        fwhm_eta_uGMT_Ext = find_fwhm(delta_eta_ext)
        fwhm_eta_uGMT = find_fwhm(np.array((self.df_test['deltaEta1'])))

        fwhm_pt_NN = find_fwhm(np.array(delta_pt_p/self.df_test['ptRecoMu1']))
        fwhm_pt_uGMT = find_fwhm((self.df_test['deltaPt1']/self.df_test['ptRecoMu1']))
        
        if self.verbose:
            
            
            print("-------------------------------------")
            print("-------------------------------------")
            print("FWHM phi uGMT",fwhm_phi_uGMT)
            print("FWHM phi uGMT-EXT",fwhm_phi_uGMT_Ext)
            print("-------------------------------------")
            print("FWHM eta uGMT",fwhm_eta_uGMT)
            print("FWHM eta uGMT-EXT",fwhm_eta_uGMT_Ext)
            print("FWHM pt uGMT",fwhm_pt_uGMT)
            print("-------------------------------------") 
            print("FWHM phi NN", fwhm_phi_NN)
            print("FWHM eta NN",fwhm_eta_NN)
            print("FWHM pt-NN",fwhm_pt_NN)
            print("-------------------------------------")
            print("-------------------------------------")
        
        if self.use_wandb:
            wandb.log({"Epoch": epoch,
                       "FWHM_phi_NN":fwhm_phi_NN,
                       "FWHM_phi_uGMT":fwhm_phi_uGMT,
                       "FWHM_phi_uGMT_EXT":fwhm_eta_uGMT_Ext,
                       
                       "FWHM_eta_NN":fwhm_eta_NN,
                       "FWHM_eta_uGMT":fwhm_eta_uGMT,
                       "FWHM_eta_uGMT_EXT":fwhm_eta_uGMT_Ext,

                       "FWHM_pt_NN":fwhm_pt_NN,
                       "FWHM_pt_uGMT":fwhm_pt_uGMT,
                       })


