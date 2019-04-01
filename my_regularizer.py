import keras
import tensorflow as tf
import six
from keras import backend as K
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object


class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class L0t1(Regularizer):
    def __init__(self, factor=0.01, sigma=0.05):
        self.factor = K.cast_to_floatx(factor)
        self.sigma = K.cast_to_floatx(sigma)

    def __call__(self, x):
        per_num=x.get_shape().as_list()[1]
        L2=tf.divide(tf.reduce_sum(tf.square(x),axis=1),per_num)
        L2_plus_sigma=L2+self.sigma
        regularization=tf.reduce_sum(tf.divide(L2,L2_plus_sigma))
        return self.factor*regularization

    def get_config(self):
        return {'factor': float(self.factor),
                'sigma': float(self.sigma)}


class L0t2(Regularizer):
    def __init__(self, factor=0.01, sigma=0.05):
        self.factor = K.cast_to_floatx(factor)
        self.sigma = K.cast_to_floatx(sigma)

    def __call__(self, x):
        per_num=x.get_shape().as_list()[1]
        L2_square=tf.square(tf.divide(tf.reduce_sum(tf.square(x),axis=1),per_num))
        sigma_square=tf.square(self.sigma)
        regularization=tf.reduce_sum(1-tf.exp(-tf.divide(L2_square,2*sigma_square)))
        return self.factor*regularization
    
    def get_config(self):
        return {'factor': float(self.factor),
                'sigma': float(self.sigma)}



class L0t3(Regularizer):
    def __init__(self, factor=0.01,sigma=1/2):
        self.factor = K.cast_to_floatx(factor)
        self.sigma = K.cast_to_floatx(sigma)

    def __call__(self, x):
        per_num=x.get_shape().as_list()[1]
        L2=tf.divide(tf.reduce_sum(tf.square(x),axis=1),per_num)
        numerator=1-tf.exp(-2*pow(L2,self.sigma))
        denominator=1+tf.exp(-2*pow(L2,self.sigma))
        regularization=tf.reduce_sum(tf.abs(tf.divide(numerator,denominator)))
        return self.factor*regularization
    
    def get_config(self):
        return {'factor': float(self.factor),
                'sigma': float(self.sigma)}



class L0t4(Regularizer):
    def __init__(self, factor=0.01, sigma=0.05):
        self.factor = K.cast_to_floatx(factor)
        self.sigma = K.cast_to_floatx(sigma)

    def __call__(self, x):
        per_num=x.get_shape().as_list()[1]
        L2=tf.divide(tf.reduce_sum(tf.square(x),axis=1),per_num)
        regularization=tf.reduce_sum(pow(L2,self.sigma))
        return self.factor*regularization
    
    def get_config(self):
        return {'factor': float(self.factor),
                'sigma': float(self.sigma)}


class L0t5(Regularizer):
    def __init__(self, factor=0.01, sigma=0.05):
        self.factor = K.cast_to_floatx(factor)
        self.sigma = K.cast_to_floatx(sigma)

    def __call__(self, x):
        per_num=x.get_shape().as_list()[1]
        L2=tf.divide(tf.reduce_sum(tf.square(x),axis=1),per_num)
        numerator=1-tf.exp(-2*(L2))
        denominator=1+tf.exp(-2*(L2))
        regularization=tf.reduce_sum(tf.abs(tf.divide(numerator,denominator)))
        return self.factor*regularization
    
    def get_config(self):
        return {'factor': float(self.factor),
                'sigma': float(self.sigma)}

    
    

    
    
    

def reg_loss(type, factor, sigma):
    if type == "type1":
        return L0t1(factor, sigma)
    elif type == "type2":
        return L0t2(factor, sigma)
    elif type == "type3":
        return L0t3(factor, sigma)
    elif type=="type4":
        return L0t4(factor, sigma)
    elif type=="type5":
        return L0t5(factor, sigma)

    
    
    


def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            'Could not interpret regularizer identifier: ' + str(identifier))