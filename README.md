# A Sliced Wasserstein Loss for Neural Texture Synthesis

This is the official implementation of  ["A Sliced Wasserstein Loss for Neural Texture Synthesis" paper](https://arxiv.org/abs/2006.07229) (to appear in CVPR 2021).

![caption paper](https://unity-grenoble.github.io/website/images/thumbnails/publication_sliced_wasserstein_loss.png)

This implementation focus on the key part of the paper: the sliced wasserstein loss.

## Requirements

### Librairies

The following libraries are required:

- Tensorflow 2
- scipy
- Matplotlib
- Numpy

To install requirements:


```setup
pip install -r requirements.txt
```

This code has been tested with Python 3.7.5 on Ubuntu 18.04.

### Pretrained vgg-19 network

A custom vgg network is used, as explained in the supplementals.
It has been modified compared to the keras standard model:

- inputs are preprocessed (including normalization with imagenet stats).
- activations are scaled.
- max pooling layers are replaced with average pooling layers.
- zero padding is remplaced with reflect padding.

## Texture generation (section 3 of the paper)

To generate a texture use the following command:

```eval
python texturegen.py [-h] [--size SIZE] [--output OUTPUT] [--iters ITERS] filename
```

The parameters are:

- iters: number of steps of l-bfgs (by default: 20). Each step is one call to scipy l-bfgs with maxfun=64.
- size: the input texture will be resized to this size (by default: 256, which resize to 256x256). If the image is not a square, it will be center cropped. The generated texture will have the same resolution.
- output: name of the output file (by default: output.jpg).
- filename: name of the input texture (only mandatory parameter).

Outputs files are:

- resized-input.jpg: input image is resized following the --size parameter. It will be exported as "resized-input.jpg" so it can be compared with the generated output.
- output file: final output file after all iterations. The name is specified by the --output tag (output.jpg by default).
- output-iterN.jpg: the intermediate result after N iterations. If there are 20 iterations, there will be 20 output images.


For instance :

```
python texturegen.py input.jpg
```

## Timing

Timing reference for 20 iterations on 256x256 resolution (which is overkill as good results appear earlier):

- On GPU (NVIDIA GTX 1080 Ti): 3min58.
- On CPU (intel i5-8600k CPU): 37min33.
