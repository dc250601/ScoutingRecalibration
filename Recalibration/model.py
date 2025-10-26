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

def make_model(hidden_neurons, k_reg, a_reg, ap_fixed,corrector=False):
    model = Sequential()
    
    # layer 1
    model.add( QDense(hidden_neurons, input_shape=(5,),
                      name='hd1',
                      kernel_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      bias_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=k_reg) )
    
    model.add( QBatchNormalization(name='bn1',
                                   beta_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   gamma_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   mean_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   variance_quantizer=quantized_bits(*ap_fixed,alpha=1)))
    
    model.add( QActivation(name='act1',
                           activation=quantized_relu(*ap_fixed),))
    
    
    # layer 2
    model.add( QDense(hidden_neurons,
                      name='hd2',
                      kernel_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      bias_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=k_reg) )
    
    # model.add( BatchNormalization(name='bn2') )
    
    model.add( QBatchNormalization(name='bn2',
                                   beta_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   gamma_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   mean_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   variance_quantizer=quantized_bits(*ap_fixed,alpha=1)))
    
    model.add( QActivation(name='act2',
                           activation=quantized_relu(*ap_fixed),))
    
    # layer 3
    model.add( QDense(hidden_neurons,                   
                      name='hd3', 
                      kernel_quantizer=quantized_bits(*ap_fixed,alpha=1), 
                      bias_quantizer=quantized_bits(*ap_fixed,alpha=1), 
                      kernel_initializer='lecun_uniform', 
                      kernel_regularizer=k_reg) )
    
    # model.add( BatchNormalization(name='bn3') )
    
    model.add( QBatchNormalization(name='bn3',
                                   beta_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   gamma_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   mean_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   variance_quantizer=quantized_bits(*ap_fixed,alpha=1)))
    
    model.add( QActivation(name='act3',               
                           activation=quantized_relu(*ap_fixed)) )
    
    # layer 4
    model.add( QDense(hidden_neurons,
                      name='hd4',
                      kernel_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      bias_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=k_reg) )
    
    # model.add( BatchNormalization(name='bn4') )
    
    model.add( QBatchNormalization(name='bn4',
                                   beta_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   gamma_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   mean_quantizer=quantized_bits(*ap_fixed,alpha=1),
                                   variance_quantizer=quantized_bits(*ap_fixed,alpha=1)) )
    
    model.add( QActivation(name='act4',                                   
                           activation=quantized_relu(*ap_fixed)) )
    
    
    # output
    if corrector:
        model.add( QDense(1,
                          name='output',
                          kernel_quantizer=quantized_bits(*ap_fixed,alpha=1),
                          bias_quantizer=quantized_bits(*ap_fixed,alpha=1),
                          kernel_initializer='lecun_uniform',
                      kernel_regularizer=k_reg) )
    else:
        model.add( QDense(3,
                      name='output',
                      kernel_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      bias_quantizer=quantized_bits(*ap_fixed,alpha=1),
                      kernel_initializer='lecun_uniform',
                      kernel_regularizer=k_reg) )
    
    model.add( Activation(name='act5',
                          activation='linear') )
    return model