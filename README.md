# Makeup Removal System with CNN and GAN




## Updated Version
To run the code, download the dataset labeled "gan_makeup_data_96", and the WGAN-GP folder. Open a terminal and cd to "src/model". Type the command:
	
python main.py --run train --use_XLA

Be careful about the dataset direction in the train_wgan_GP.py.

### Results
Before actually inputting images to the GAN, we first tried to use random noises as inputs to test the
capability of the GAN. One of the results of noisy input is showing in Figure 1(a), with the input to
the generator as random noise and the input to the discriminator as without-makeup images.

The results of the system are showing in Figure 1(b). The first row contains the with-makeup images
that were input into the generator. The second row shows the makeup-removed images generated from
the generator. The final row contains the ground truth images of the same people without makeup.
The generated images are different with the ground truth image in poses and facial expressions,
because the pair of photos of a same person in the dataset were taken in different poses and facial
expressions.

| [![](https://github.com/Oceanland-428/Makeup-Removal-System-with-CNN-and-GAN/blob/master/noisy_results.png)]()  | [![](https://github.com/Oceanland-428/Makeup-Removal-System-with-CNN-and-GAN/blob/master/removal_results.png)]() |
|:---:|:---:|
| Figure 1(a) | Figure 1(b) |

## Proposal Version

This project is a makeup removal deep learning system. There are several application areas of such a system. For example, a front door face authentication camera should be able to recognize the users regardless the level of the makeup. In addition, the system is useful for social network apps since facial cosmetics can sometimes be deceivable.

Dataset of this project consists of total 2600 images of 1300 different people from five separate datasets (FAM, MIFS, VMU, MIW, YMU). Each person has two images: one with makeup and the other one without. 

The system contains three pretrained neural networks and one self-build neural network, as illustrated in Figure 2. Before building our own network, we need to find three pretrained networks, A, B and C (B is a free gift if we have A and C, so actually we only need to have two pretrained networks). A is used to detect whether a person is with or without makeup. B is used to detect whether an image is a valid person’s photo. C is a face recognition network, which provides encodings of the input face image. The network on the bottom of Figure 1 is a self-build network. The input is a person’s photo with makeup. The network then modifies the photo and outputs a candidate image. This image then goes through the three pretrained networks. If the output of A and B is without makeup and valid, respectively, and the encodings from C are similar to the encodings of the original person’s photo, then the candidate image is output from the network. If any of the above condition is not satisfied, the image will return to the network and go through the process again.

One or more pretrained networks will be derivations of VGG16, with some modification on the last few fully connected layers of CNN using our training examples. There are similar but different implementations existing in Github: CycleGan, Pix2Pix.

<p align="center">
  <img width="766" height="466" src="https://github.com/Oceanland-428/Makeup-Removal-System-with-CNN-and-GAN/blob/master/System_Arch.png">
</p>
<p align="center">
  Figure 2
</p>

#
The programs are implemented in Tensorflow, Python 2.7

Dataset website/paper:

MIFS, VMU, MIW, YMU: http://www.antitza.com/makeup-datasets.html

FAM: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6638073

Resources for reference:

CycleGan: http://github.com/junyanz/CycleGAN

Pix2Pix: http://github.com/phillipi/pix2pix

Special thanks to our TA Olivier and his finetune tutorial:

Finetune: http://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/

### References:

[1] Dantcheva, A., Chen, C., & Ross, A. (2012) Can Facial Cosmetics Affect the Matching Accuracy of Face
Recognition Systems? 5th IEEE International Conference on Biometrics: Theory, Applications and Systems

[2] Wang, S., & Fu, Y. (2016). Face Behind Makeup. Association for the Advancement of Artificial Intelligence,
pp. 58-64.

[3] Jun-Yan, Z., Richard, Z., Deepak, P., Trevor, D., Alexei, E., Oliver, W. & Eli, S. (2017) Toward Multimodal
Image-to-Image Translation. Computer Vision and Pattern Recognition

[4] Ishaan, G., Faruk, A., Martin, A., Vincent, D. & Aaron C. (2017) Improved Training of Wasserstein GANs.
Learning

[5] Yi, L., Lingxiao, S., Xiang, W., Ran, H. & Tieniu, T. (2017) Anti-Makeup: Learning A Bi-Level Adversarial
Network for Makeup-Invariant Face Verification. Computer Vision and Pattern Recognition

[6] Jun-Yan, Z., Taesung, P., Phillip, I. & Alexei, E. (2018). Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks. Computer Vision and Pattern Recognition

[7] Chen, C., Dantcheva, A., & Ross, A. (2013) Automatic Facial Makeup Detection with Application in Face
Recognition. 6th IAPR International Conference on Biometrics

[8] Chen, C., Dantcheva, A., & Ross, A. (2016) An Ensemble of Patch-based Subspaces for Makeup-Robust
Face Recognition. Information Fusion Journal, Vol.32, pp. 80-92

[9] Chen, C., Dantcheva, A., Swearingen, T., & A. Ross, (2017) Spoofing Faces Using Makeup: An Investigative
Study. 3rd IEEE International Conference on Identity, Security and Behavior Analysis

[10] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks forsemantic segmentation. In CVPR,
pages 3431-3440, 2015. 2, 3, 6

[11] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial
networks. In CVPR, 2017. 2, 3, 5, 7, 20
