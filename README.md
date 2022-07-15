#### Gustavo Contini Torres - Nro USP 5373002
#### Jose Alexandre F Silva - Nro USP 11908649
#
# Project Title: Feature Extraction Analysis on Electronic Waste
#
# Abstract
The purpose of this project is to analyze a set of images related to Electronic Waste, by working with Feature Extraction techniques, namely: Binary Robust Invariant Scalable Keypoints (BRISK), Gray Level Co-occurrence Matrix (GLCM) Haralick Features and Local Binary Patterns (LBP). Then the extracted filters will be clustered and Multidimensional Projections (PCA and t-SNE) are applied to produce point cloud charts in the attempt to visually interpret the descriptors obtained. In addition, a Bag of Visual Words will be defined and used as input for image classification and content-based image retrieval.

This procedure is repeated to synthetic images created by data augmentation and background/foreground segmentantion methods. Finally, we compare the accuracy of the classification models developed for each combination of image dataset and feature extraction technique. Also, we comment on the results and interpretaion of image retrieval, closing this report with our understanding of the lessons learned from this project.

The inputs were obtained by taking photos of the Scrap / Waste Disposal operation of a large IT company, which allowed us to get a variety of electronic waste images, like Printed Circuit Boards, Connectors, Cables and Wires, Batteries, Screens, Magnetic Tapes, Metals, etc.

# 1. Main Objective
Analyze a set of images related to Electronic Waste by extracting Texture Features and then using these features in two tasks: image classification and image retrieval.

# 2. Description of the Input Images:
As shown in the examples below, the images refer to Electronic Waste (Printed Circuit Boards, Connectors, Cables and Wires, Batteries, Screens, Magnetic Tapes, Metals, etc.) and they were obtained by taking photos of the Scrap / Waste Disposal operation of a large IT company. Given the main objective of the project and the methods chosen, we don't propose the production of output images. In the other hand, we apply dimension reduction techniques and plot point cloud charts to try to create a visualization of the features extracted from the images.

# 3. Description of the Steps to reach the Objective:
We will work with three feature extraction methods: Gray-Level Co-Occurrence Matrix (GLCM), Local Binary Patterns (LBP) and Binary Robust Invariant Scalable Keypoints (BRISK).

GLCM is a statistical method of examining texture that considers the spatial relationship of pixels in the Gray-Level Co-Occurrence matrix, also known as the gray-level spatial dependence matrix. The GLCM functions characterize the texture of an image by calculating how often pairs of pixel with specific values and in a specified spatial relationship occur in an image, creating a GLCM, and then extracting statistical measures from this matrix.

LBP is a texture descriptor used for the property of high discrimination power. LBP labels each pixel in an image by comparing the gray level with the neighboring pixels and then assigning a binary number. A value of unity is assigned to the neighbors with gray level greater than the center pixel in a predefined patch, otherwise a value of zero. A binary number is then obtained and assigned to the center pixel.

BRISK is a feature point detection and description algorithm with scale and rotation invariance. It constructs the feature descriptor of the local image through the gray scale relationship of random point pairs in the neighborhood of the local image, and obtains the binary feature descriptor.

For the analysis and interpretation of the results, we will apply Distance Functions (Euclidean, Cosine, Manhattan) and Multidimensional Projection (t-SNE, MDS or PCA). Also, the idea is to define a Bag of Visual Words for content-based image retrieval in order to identify the best combination of features.

# 4. Initital Code (testing basic functions): 
https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Feature%20Extraction%20Analysis%20on%20Electronic%20Waste.ipynb


# Images - Examples:

## Battery
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Battery100.JPG" width="300">

## Cable
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Cable%20and%20Wire41.JPG" width="300">

## Connector
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Connector4.JPG" width="300">

## Magnetic Tape
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Magnetic%20Tape8.JPG" width="300">

## Printed Circuit Board
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Printed%20Circuit%20Board139.JPG" width="300">

## Screen
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Tube%20and%20Screen33.JPG" width="300">

