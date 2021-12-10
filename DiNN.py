from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf

import numpy as np



class GraphConstrained(Layer):
    """
    A forward passing layer with connection to previous layer specified
    by a binary matrix of shape '(Output_dim, Input_dim)'.
    
    
    **Input**

    - Tensor of shape `([batch], Input_dim)`;

    **Output**

    - Tensor of shape `([batch], Output_dim)`;

    **Arguments**

    - `activation`: activation function to use;
    - `adj`: adjacency matrix;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(self,
                 adj,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.adj = tf.constant(adj, dtype = tf.float32)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 1
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=self.adj.shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.adj.shape[-1],),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.built = True

    def call(self, inputs):
        
        features = inputs
        fltr = self.kernel * self.adj

        # Convolution
        output = tf.matmul(features, fltr)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'adj': self.adj,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        config.update(super().get_config())
        return config

    
class FG_Squircular_to_disk(Layer):
    '''
    A layer that maps coordinate from a square to a circle disk
    with FG-Squircular Mapping
    
    input: (x,y) coordinates
    
    output: (u,v) mapped coordinates
    '''
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        
    def call(self, inputs):
        numerator = tf.sqrt(tf.math.square(inputs[:,0])+tf.math.square(inputs[:,1]) - tf.math.square(inputs[:,0] * inputs[:,1]))
        denominator = tf.sqrt(tf.math.square(inputs[:,0])+tf.math.square(inputs[:,1]))
        scaling = self.scale * tf.reshape(numerator/denominator, [-1,1])
        return inputs * scaling