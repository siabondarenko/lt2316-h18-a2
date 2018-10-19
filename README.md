# LT2316 H18 Assignment 2

Git project for implementing assignment 2 in [Asad Sayeed's](https://asayeed.github.io) machine learning class in the University of Gothenburg's Masters
of Language Technology programme.

## Your notes

Running: 

python3 train.py -P A (-m maxinstances) modelfile cat1 cat2 cat3...

python3 test.py -P A (-m maxinstances) modelfile cat1 cat2 cat3...


The model's architecture consists of a sequence of conv2d layers followed by maxpooling2d layers in the encoding part,
as well as conv2dtranspose layers followed by upsampling2d layers in the decoding part.

As far as I understand, mean squared error seems to be the most appropriate loss metric for autoencoders, while accuracy is usually omitted. 
I've replicated both of these settings in my model.

From what I've noticed in various tutorials the batch setting for CNN image autoencoders usually seems to be either 16, or 32.
In my case changing it didn't really affect the loss (which seems to be around 0.073), so in my script it's set to 16.

Steps per epoch are set to "number of samples" / "batch" according to the recommendations that I've seen. 
Epochs are set to 20 because I wasn't noticing any loss reduction with additional epochs in the current version of the model.

autoencoder.model has been trained on 10926 images with: python3 train.py -P A 'autoencoder.model' 'tv' 'oven' 'laptop'

Both training and testing scripts output the shape of the predictions array since that's as far as I've managed to get with this assignment.




