import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib.request
import argparse

parser = argparse.ArgumentParser(description='Texture generation')
parser.add_argument('filename', help='filename of the input texture')
parser.add_argument('--size', default=256, type=int, help='resolution of the input texture (it will resize to this resolution)')
parser.add_argument('--output', default='output.jpg', help='name of the output file')
parser.add_argument('--iters', default=20, type=int, help='number of steps')
args = parser.parse_args()


SIZE = args.size
INPUT_FILE = args.filename
OUTPUT_FILE = args.output
NB_ITER = args.iters

print(f'Launching texture synthesis from {INPUT_FILE} on size {SIZE} for {NB_ITER} steps. Output file: {OUTPUT_FILE}')

def decode_image(path, size=SIZE):
    """ Load and resize the input texture """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    max_size = img.shape[0]
    if max_size > img.shape[1]: max_size = img.shape[1]

    img = tf.image.resize_with_crop_or_pad(img, max_size, max_size)
    img = tf.image.resize(img, [size, size]) 
    img = tf.image.convert_image_dtype(img, tf.float32) / 255

    return img[None]


class Slicing(tf.keras.layers.Layer):
    """ Slicing layer: computes projections and returns sorted vector """
    def __init__(self, num_slices):
        super().__init__()
        # Number of directions
        self.num_slices = num_slices
        self.flatten_layer = tf.keras.layers.Flatten()

    def update_slices(self):
        """ Update random directions """
        # Generate random directions
        self.directions = tf.random.normal(shape=(self.num_slices, self.dim_slices))
        # Normalize directions
        norm = tf.reshape( K.sqrt( K.sum( K.square(self.directions), axis=-1 )), (self.num_slices, 1))
        self.directions = tf.divide(self.directions, norm)

    def build(self, input_shape):
        self.dim_slices = input_shape[-1]
        self.update_slices()

    def call(self, input):
        """ Implementation of figure 2 """
        tensor = tf.reshape(input, (tf.shape(input)[0], -1, tf.shape(input)[-1]))
        # Project each pixel feature onto directions (batch dot product)
        sliced = self.directions @ tf.transpose(tensor, perm=[0,2,1])
        # Sort projections for each direction
        sliced = tf.sort(sliced)
        
        return self.flatten_layer(sliced)

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # A customized vgg is used as explained in the supplementals.
    vgg = keras.models.load_model('vgg_customized.h5')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model

class ExtractorVggModel(tf.keras.Model):
    """ Extract stats using a pretrained vgg and return slices vectors"""
    def __init__(self, layers):
        super().__init__()
        self.vgg =  vgg_layers(layers)
        self.vgg.trainable = False    

        self.slicing_losses = [Slicing(num_slices=l.shape[-1]) for i, l in enumerate(self.vgg.outputs)]

    def update_slices(self):
        for slice_loss in self.slicing_losses:
            slice_loss.update_slices()

    def call(self, inputs):
        outputs = self.vgg(inputs)
        outputs = [self.slicing_losses[i](output)
                        for i, output in enumerate(outputs)]

        return outputs

    

def loss_and_grad(image, *args):
    """ Return loss and grad for a step. Called by lbfgs optimize  """
    image_tf = tf.constant(np.reshape(image, (1,SIZE,SIZE,3)), dtype='float32')
    args = args[0]
    extractor = args['extractor']
    targets = args['targets']

    with tf.GradientTape() as tape:
        tape.watch(image_tf) 
        outputs = extractor(image_tf)
        #L2 between the sorted slices (generated image vs target texture)
        losses =[tf.reduce_mean((output-targets[i])**2) 
                           for i, output in enumerate(outputs)]
        loss = tf.add_n(losses)
    
    grad_raw = tape.gradient(loss, image_tf)    
    grad = np.reshape(grad_raw.numpy(), (SIZE*SIZE*3,))
    loss = np.reshape(loss.numpy(), (1))  

    return loss.astype('float64'), grad.astype('float64')


def fit(nb_iter, texture, extractor):
    targets = extractor(texture)
  
    # Image initialization  
    image = np.zeros((1,SIZE,SIZE,3))
    image = image + tf.reduce_mean(texture, axis=(1, 2))[None, None]
    image = image + (tf.random.normal((1, SIZE,SIZE, 3))*1e-2)
        
    
    for i in range(nb_iter):
        arg_opt = {'extractor': extractor, 'targets':targets}

        image, loss, info = fmin_l_bfgs_b(func=loss_and_grad, args=[arg_opt], x0=image, maxfun=64, pgtol=0.0, factr=0.0)
        image = np.clip(image, 0, 1)
        print(f'iter {i+1} loss {loss}')
        
     
        # Change random directions (optional)
        extractor.update_slices()
        targets = extractor(texture)

        #export image at the current iteration
        image = np.reshape(image, (SIZE, SIZE, 3)).astype('float32')
        plt.imsave(f'output-iter{i+1}.jpg', image)
            
    return image


############################ MAIN ############################
# VGG layers used for the loss
layers = ['block1_conv1',
                'block1_conv2',
                'block2_conv1',
                'block2_conv2',
                'block3_conv1', 
                'block3_conv2',
                'block3_conv3',
                'block3_conv4',
                'block4_conv1', 
                'block4_conv2',
                'block4_conv3',
                'block4_conv4',
                'block5_conv1',
                'block5_conv2'
               ]


extractor = ExtractorVggModel(layers)

texture_reference = decode_image(INPUT_FILE, size=SIZE)
# export of the resized input: reference to compare against the generated texture
plt.imsave('resized-input.jpg', texture_reference.numpy()[0])

output_image = fit(NB_ITER, texture_reference, extractor)
plt.imsave(OUTPUT_FILE, output_image)   
