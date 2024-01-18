
# Neural Style Transfer

The objective is to implement a style transfer algorithm using deep learning techniques

## Background

Style transfer is a technique that allows the style of one image to be transferred to another image while preserving the content of the original image. This technique has been widely used in the field of computer vision and has many applications, such as artistic style transfer, image colorization, and image enhancement.

## How Does Neural Style Transfer Work?

Neural Style Transfer(NST) employs a pre-trained Convolutional Neural Network with added loss functions to transfer style from one image to another and synthesize a newly generated image with the features we want to add.

Style transfer works by activating the neurons in a particular way, such that the output image and the content image should match particularly in the content, whereas the style image and the desired output image should match in texture, and capture the same style characteristics in the activation maps.

These two objectives are combined in a single loss formula, where we can control how much we care about style reconstruction and content reconstruction.

Here are the required inputs to the model for image style transfer:

* A Content Image : an image to which we want to transfer style to
* A Style Image : the style we want to transfer to the content image
* An Input Image (generated) : the final blend of content and style image

![NST Working](https://github.com/wolfblunt/StyleTransfer/blob/master/Images/styleTransfer.png)

## Neural Style Transfer Structure

![NST Structure](https://github.com/wolfblunt/StyleTransfer/blob/master/Images/nstStructure.jpg)

Neural Style Transfer(NST) uses a pre-trained model trained on ImageNet- VGG in TensorFlow. Images themselves make no sense to the model. These have to be converted into raw pixels and given to the model to transform it into a set of features, which is what Convolutional Neural Networks are responsible for.

Thus, somewhere in between the layers, where the image is fed into the model, and the layer, which gives the output, the model serves as a complex feature extractor. All we need to leverage from the model is its intermediate layers, and then use them to describe the content and style of the input images.

The input image is transformed into representations that have more information about the content of the image, rather than the detailed pixel value.

The features that we get from the higher levels of the model can be considered more related to the content of the image.

To obtain a representation of the style of a reference image, we use the correlation between different filter responses.

## Steps
1. Import and configure modules
2. Visualize the input
3. Define content and style representations
4. Build the model
5. Calculate Style using Gram Matrix
6. Extract style and content
7. Run the Optimization
8. Finally, save the result



## Libraries Used
* `matplotlib.pyplot`: For creating visualizations.
* `numpy`: For numerical operations.
* `PIL (Python Imaging Library) / Pillow` : used for opening, manipulating, and saving various image file formats. In neural style transfer, it can be used to load and process input images.
* `functools` : The functools module provides higher-order functions and operations on callable objects. It may be used for functional programming aspects in the code.
* `tensorflow (tf)` : used for building and training deep learning models. It provides efficient operations for numerical computations on tensors, which are multi-dimensional arrays. The neural style transfer algorithm often involves optimizing an image to minimize a loss function, and TensorFlow is suitable for such optimization tasks.
* `IPython.display` : used for interactive computing in Python. In the context of neural style transfer, it might be used to display images or visualizations within an IPython (Interactive Python) environment.

-----


#### Notebook Link
-----

https://colab.research.google.com/drive/12299jrc0wBTnJ5rOK9a_CG1luBpJBo9O?usp=sharing

-----

### Code credit

Code credits for this code go to [Aman Khandelwal](https://github.com/wolfblunt)