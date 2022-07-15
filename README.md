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

# 2. Description of the Input Images
As shown in the examples below, the images refer to Electronic Waste (Printed Circuit Boards, Connectors, Cables and Wires, Batteries, Screens, Magnetic Tapes, Metals, etc.) and they were obtained by taking photos of the Scrap / Waste Disposal operation of a large IT company. Given the main objective of the project and the methods chosen, we don't propose the production of output images. In the other hand, we apply dimension reduction techniques and plot point cloud charts to try to create a visualization of the features extracted from the images.

## Images - Examples:

### Battery
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Battery100.JPG" width="300">

### Cable
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Cable%20and%20Wire41.JPG" width="300">

### Connector
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Connector4.JPG" width="300">

### Magnetic Tape
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Magnetic%20Tape8.JPG" width="300">

### Printed Circuit Board
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Printed%20Circuit%20Board139.JPG" width="300">

### Screen
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/all/Tube%20and%20Screen33.JPG" width="300">

# 3. Description of the Steps to reach the Objective
To achieve the object of this project we worked with three feature extraction methods: Gray-Level Co-Occurrence Matrix (GLCM), Local Binary Patterns (LBP) and Binary Robust Invariant Scalable Keypoints (BRISK).

GLCM is a statistical method of examining texture that considers the spatial relationship of pixels in the Gray-Level Co-Occurrence matrix, also known as the gray-level spatial dependence matrix. The GLCM functions characterize the texture of an image by calculating how often pairs of pixel with specific values and in a specified spatial relationship occur in an image, creating a GLCM, and then extracting statistical measures from this matrix. We used haralick feature extraction implemented in the Mahotas package for Python. Mahota's implementations contemplates the first 13 features from Haralick's original work. They are: (1) Angular Second Moment, (2) Contrast, (3) Correlation, (4) Variance, (5) Inverse Difference Moment, (6) Sum Average, (7) Sum Variance, (8) Sum (9) Entropy, (10) Difference Variance, (11) Difference Entropy and (12 and 13) Information Measures of Correlation.

LBP is a texture descriptor that takes the surrounding points of a central pixel and tests whether they are greater than or less than the central point (i.e. gives a binary result). By doing so it creates both uniform and non-uniform patterns indicating changes or regions of similar textures. These patterns then have their distribuitions analysed via their histogram. We used LBP implementation available in the scikit-image package for Python.

BRISK is a feature point detection and description algorithm with scale and rotation invariance. It constructs the feature descriptor of the local image through the gray scale relationship of random point pairs in the neighborhood of the local image, and obtains the binary feature descriptor. BRISK was proposed by S. Leutenegger, M. Chli and R. Y. Siegwart in 2011 as a novel method for keypoint detection, description and matching, delivering high quality performance as in state-of-the-art algorithms, such as SIFT and SURF, albeit at a dramatically lower computational cost.

After applying these techniques to extract features from the images, we use the k-means algorithm to create clusters for the descriptors. We try a range of values from 25 to 1,000 for the number of clusters and decide the best option by comparing their Silhouette Scores. Then we apply two Multidimension Projection methods to reduce the high cardinallity of the data and then we plot 2D point cloud charts of the two main components obtained from PCA and t-SNE.

Results are discussed in detail in the next section, but clustering and visualizing point clouds of the main components didn't produce good outputs. So we decided to use the extracted features as inputs in Machine Learning classification algorithms to better explore their potential to be used to classify other images. Thus we used six algorithms (namely Logistic Regression, Multi-Layer Perceptron, Support Vector Machine, Decision Tree, Random Forest and eXtreme Gradient Boosting) in a cross-validation training procedure with 5 Stratified Folds and measured the mean accuracy of each model to proceed with the one with the highest value.

We then trained the best model for each set of features (extracted by the three different methods) and used these models to predict the labels (or classes) of images from a different set (test dataset) and, again, measured the accuracy of the models, but not only the overall accuracy, but also the individual measure for each class.

This whole procedure of extracting the features, clustering and visualizing their descriptors, training machine learning models and using them to predict the classes of new images, was replicated to three other sets of images. These other sets of images consist in synthetic reproductions of the original ones, obtained by the application of data augmentation and background/foreground segmentation.

The data augmentation was done using the Keras package on TensorFlow platform. With the objective of balancing the number of elements in each class, we created new images by applying small variations in the original ones' dimensions, such as width and height changes, 90 degrees rotation, brithness adjustments and horizontal flip. 

As for the background/foreground segmentation, we used the U2-Net pre-trained Neural Network and applied it to our set of images. U2-Net is a two-level nested Neural Network using an U-structure architecture that is designed for salient object detection (SOD). The architecture allows the network to go deeper, attain high resolution, without significantly increasing the memory and computation cost. This is achieved by a nested U-structure: on the bottom level, with a novel ReSidual U-block (RSU) module, which is able to extract intra-stage multi-scale features without degrading the feature map resolution; on the top level, there is a U-Net like structure, in which each stage is filled by a RSU block.

Finally, we combined these two techniques and created the third dataset of new synthetical images. The outputs and obtained results for the multi classification task are discussed in the next section.

### José, inclua as informações sobre image retrieval aqui.

# 4. Obtained Results and Discussion


https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Feature%20Extraction%20Analysis%20on%20Electronic%20Waste.ipynb

