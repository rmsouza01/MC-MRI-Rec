# MR-reconstruction-challenge

## General description
The Calgary-Campinas dataset team is launching a magnetic resonance (MR) brain imaging reconstruction challenge.  The challenges has two independent components:

- Single-channel coil reconstruction challenge
- Multi-channel coil reconstruction challenge

Teams are free to decide if they want to participate of just one  or both components. One of the main reasons we are launching this challenge is the lack of a good brain MR raw dataset for benchmark purposes.  Data provided here were collected as part of the ongoing Calgary Normative Study. On a yearly basis, the Calgary-Campinas team will assess the possibility/necessity of increasing the amount of data provided as part of this challenge and potentially include prospectively undersampled data.

## Single-channel Coil Data (~8 GB uncompressed)

We are providing  35 fully-sampled (+10 for testing)  T1-weighted MR datasets acquired on a clinical MR scanner (Discovery MR750; General Electric (GE) Healthcare, Waukesha, WI) . Data were acquired with a 12-channel imaging coil. The multi-coil k-space data was reconstructed using vendor supplied tools (Orchestra Toolbox; GE Healthcare). Coil sensitivity maps were normalized to produce a single complex-valued image set that could be back-transformed to regenerate complex k-space samples. You can see this data as a 3D acquisition in which the inverse Fourier Transform was applied on the readout direction, which essentially allows you to treat this problem as a 2D problem while at the same time undersampling on two directions (slice encoding and phase encoding). The matrix size is 256 x 256.

We are providing the train and validation sets. The data are split as follows:
- Train: 25 subjects - 4,524 slices
- Validation: 10 subjects - 1,700 slices
- Test: 10 subjects - 1,700 slices (not provided, it will be used to test the model you submit)

This single-coil data is meant as proof of concept for assessing the ability of using Deep Learning for MR reconstruction. If you already have experience with that, we suggest that you go straight to the multi-channel coil reconstruction challenge, which is a more realistic scenario.

## Multi-channel Coil Data (~55 GB uncompressed)

We are providing 60 T1-weighted MR datasets acquired on a clinical MR scanner (Discovery MR750; GE Healthcare, Waukesha, WI). Data were acquired with a 12-channel imaging coil. The inverse fast Fourier transform (iFFT) was applied to the k-space data in the readout direction. Similarly, to the single-channel coil challenge, this allows you to treat this problem as a 2D problem while at the same time undersampling on two directions (slice encoding and phase encoding). The acquisition matrix size for each channel was 256×218×170 (with a few exceptions). The reference images were reconstructed by taking the channel-wise iFFT of the collected k-spaces and combining the outputs through the conventional sum of squares algorithm.

## Download

The dataset can be downloaded from the [Calgary-Campinas Dataset website](https://sites.google.com/view/calgary-campinas-dataset/home?authuser=0).

## Code

This repository has code describing the dataset parameters, how to load and reconstruct the images, and baseline reconstruction models.

## Questions?
Contact: roberto.medeirosdeso@ucalgary.ca


## Contact information:
roberto.medeirosdeso@ucalgary.ca

Updated: 18 July 2019
