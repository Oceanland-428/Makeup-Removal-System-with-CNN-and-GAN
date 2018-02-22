# Makeup Removal System with CNN and GAN

This project is a makeup removal deep learning system. There are several application areas of such a system. For example, a front door face authentication camera should be able to recognize the users regardless the level of the makeup. In addition, the system is useful for social network apps since facial cosmetics can sometimes be deceivable.

Dataset of this project consists of total 2600 images of 1300 different people from five separate datasets (FAM, MIFS, VMU, MIW, YMU). Each person has two images: one with makeup and the other one without. 

The system contains three pretrained neural networks and one self-build neural network, as illustrated in Figure 1. Before building our own network, we need to find three pretrained networks, A, B and C (B is a free gift if we have A and C, so actually we only need to have two pretrained networks). A is used to detect whether a person is with or without makeup. B is used to detect whether an image is a valid person’s photo. C is a face recognition network, which provides encodings of the input face image. The network on the bottom of Figure 1 is a self-build network. The input is a person’s photo with makeup. The network then modifies the photo and outputs a candidate image. This image then goes through the three pretrained networks. If the output of A and B is without makeup and valid, respectively, and the encodings from C are similar to the encodings of the original person’s photo, then the candidate image is output from the network. If any of the above condition is not satisfied, the image will return to the network and go through the process again.

One or more pretrained networks will be derivations of VGG16, with some modification on the last few fully connected layers of CNN using our training examples. There are similar but different implementations existing in Github: CycleGan, Pix2Pix.

<p align="center">
  <img width="766" height="466" src="https://github.com/Oceanland-428/Makeup-Removal-System-with-CNN-and-GAN/blob/master/System_Arch.png">
</p>
<p align="center">
  Figure 1
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

1. A. Dantcheva, C. Chen, A. Ross, "Can Facial Cosmetics Affect the Matching Accuracy of Face Recognition Systems?," Proc. of 5th IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS), (Washington DC, USA), September 2012.

2. C. Chen, A. Dantcheva, A. Ross, "Automatic Facial Makeup Detection with Application in Face Recognition," Proc. of 6th IAPR International Conference on Biometrics (ICB), (Madrid, Spain), June 2013.

3. C. Chen, A. Dantcheva, A. Ross, "An Ensemble of Patch-based Subspaces for Makeup-Robust Face Recognition," Information Fusion Journal, Vol. 32, pp. 80 - 92, November 2016.

4. C. Chen, A. Dantcheva, T. Swearingen, A. Ross, "Spoofing Faces Using Makeup: An Investigative Study," Proc. of 3rd IEEE International Conference on Identity, Security and Behavior Analysis (ISBA), (New Delhi, India), February 2017.

5. Hu, J.; Ge, Y.; Lu, J.; and Feng, X. 2013. Makeup-robust face verification. In International Conference
on Acoustics, Speech and Signal Processing, 2342–2346.

6. Zhu, Jun-Yan, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired image-to-image translation using cycle-consistent adversarial networks." arXiv preprint arXiv:1703.10593 (2017).
