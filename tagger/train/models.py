"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D


def baseline(inputs_shape, inputs_global_shape, output_shape, bits=9, bits_int=2, alpha_val=1):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    inputs        = tf.keras.layers.Input(shape=inputs_shape       , name='model_input')
    inputs_global = tf.keras.layers.Input(shape=inputs_global_shape, name='model_global_input')

    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)
    
    #First Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)

    # now merge the deepsets with the global
    main = tf.keras.layers.Concatenate(axis=1)([main, inputs_global])

    main = QDense(32, name='Dense_1_main', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1_main')(main)

    main = QDense(16, name='Dense_2_main', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2_main')(main)

    main = QDense(output_shape[0], name='Dense_3_main', **common_args)(main)
    main = Activation('softmax', name='main_output')(main)

    
    #Define the model using both branches
    model = tf.keras.Model(inputs = [inputs, inputs_global], outputs = [main])

    print(model.summary())

    return model
