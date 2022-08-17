#######################################################################
# Implementation of 
# A Sliced Wasserstein Loss for Neural Texture Synthesis
# Heitz et al., CVPR 2021
#######################################################################

import numpy as np
import torch
import imageio

#######################################################################
# scaling factor of the optimized texture 
# wrt the example texture
#######################################################################
SCALING_FACTOR = 1


#######################################################################
# Load example texture 
#######################################################################
FILE_PATH = './style.jpg'


def saveImage(filename, image):
    imageTMP = np.clip(image * 255.0, 0, 255).astype('uint8')
    imageio.imwrite(filename, imageTMP)

def loadImage(filename):
    image = imageio.imread(filename).astype("float32")[:, :, 0:3] / 255.0
    image = image[np.newaxis, ...]
    return image

image_example = loadImage(FILE_PATH)
image_example = np.swapaxes(image_example, 1, 3)
image_example = torch.from_numpy(image_example)
image_example = image_example.to(torch.device("cuda:0"))

# Make sure the size is a factor of 8 
# because we go up to block4 in VGG19 where the size is divided by 8
image_example = image_example[:, :, 0:(image_example.shape[2]//8)*8, 0:(image_example.shape[3]//8)*8]


#######################################################################
# Load pretrained VGG19
#######################################################################

class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        return [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]

vgg = VGG19().to(torch.device("cuda:0"))
vgg.load_state_dict(torch.load("vgg19.pth"))


#######################################################################
# Initialize optimized texture
#######################################################################

image_optimized = torch.mean(image_example, dim=(2,3), keepdim=True) + 0.01 * torch.randn(1, 3, SCALING_FACTOR * image_example.shape[2], SCALING_FACTOR * image_example.shape[3]).to(torch.device("cuda:0"))
image_optimized = torch.nn.parameter.Parameter(image_optimized)


#######################################################################
# LBFGS optimization with the slicing loss
#######################################################################

optimizer = torch.optim.LBFGS([image_optimized], lr=1, max_iter=64, tolerance_grad=0.0)

def slicing_loss(image_generated, image_example):
    
    # generate VGG19 activations
    list_activations_generated = vgg(image_generated)
    list_activations_example   = vgg(image_example)
    
    # iterate over layers
    loss = 0
    for l in range(len(list_activations_example)):
        # get dimensions
        b = list_activations_example[l].shape[0]
        dim = list_activations_example[l].shape[1]
        n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
        # linearize layer activations and duplicate example activations according to scaling factor
        activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
        activations_generated = list_activations_generated[l].view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
        # sample random directions
        Ndirection = dim
        directions = torch.randn(Ndirection, dim).to(torch.device("cuda:0"))
        directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
        # project activations over random directions
        projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
        projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
        # sort the projections
        sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
        sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
        # L2 over sorted lists
        loss += torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
    return loss

# LBFGS closure function
def closure():
    optimizer.zero_grad()
    loss = slicing_loss(image_optimized, image_example)
    loss.backward()
    return loss

# optimization loop
for iteration in range(64):
    tmp = image_optimized.detach().cpu().clone().numpy()
    tmp = np.swapaxes(tmp, 1, 3)
    saveImage('optimized_texture_'+str(iteration)+'_iterations.png', tmp[0, ...])
    print(iteration)
    optimizer.step(closure)
    

