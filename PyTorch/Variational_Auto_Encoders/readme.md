<h1> Variational Auto Encoders </h1>

<h2> AutoEncoders </h2>

Autoencoders are special type of encoder decoder neural network where on one side you feed your data and on the other side you want to reconstruct the same data , in the middle you try to compress the data to a vector **z** where **len(z) << len(data)** . A typical autoencoder looks like this 

![autoencoder](https://wiki.tum.de/download/attachments/25007151/autoencoder_structure.png?version=2&modificationDate=1485704320100&api=v2)
![autoencoder](https://cdn-images-1.medium.com/max/703/0*yGqTBMopqHbR0fcF.)

This autoencoders are good for compression of data , but not in the case when we cant to generate new data from our training images or generate data which are completely new and never seen in the training examples , but somewhere related to our training images.

More formally we can define it as we want to build a generative model , which on trained on samples can learn and produce new samples which are from the training samples distribution, so our goal is to not only learn a netowork which can compress the training images and reconstruct it back but also which can learn the distribution which the training samples follow.

<h2> Variational Autoencoders </h2>

So in a variational autoencoder our task to is find out the distribution of our training samples and furthur produce samples from that distribution.

![image](https://camo.githubusercontent.com/74620840800d49e0e3f0fb97db950212f61ec596/687474703a2f2f6b766672616e732e636f6d2f636f6e74656e742f696d616765732f323031362f30382f7661652e6a7067)

Here we need to minimize two losses , One the **log liklihood loss** and the other is the **KL Divergence** , The first will ensure that we are able to reconstruct our training samples and the second ensures that we can find out the distribution of the training sample.

![image.png](attachment:image.png)

Here we use a CNN as an Encoder with layers and relu as activation and in Decoder we use Billinear Upsample. All the images are resized to 150x150.


# Results

## Reconstructions

#### Epoch 28
![recon](results/reconstruction_28.png)
#### Epoch 20
![recon](results/reconstruction_20.png)
#### Epoch 10
![recon](results/reconstruction_10.png)
#### Epoch 2
![recon](results/reconstruction_2.png)

## Generations

#### Epoch 28
![gen](results/sample_28.png)
#### Epoch 20
![gen](results/sample_20.png)
#### Epoch 10
![gen](results/sample_10.png)
#### Epoch 2
![gen](results/sample_2.png)

## Generation Epoch 1 to Epoch 28

![gen](results/generation.gif)