#### Gustavo Contini Torres - Nro USP 5373002
#### Jose Alexandre F Silva - Nro USP 11908649
#
# Project Title: Feature Extraction Analysis on Electronic Waste
#
# Abstract
The purpose of this project is to analyze a set of images related to Electronic Waste by working with Feature Extraction techniques, namely: Binary Robust Invariant Scalable Keypoints (BRISK), Gray Level Co-occurrence Matrix (GLCM) Haralick Features and Local Binary Patterns (LBP). Then, the extracted features were clustered and visually represented with Multidimensional Projections (PCA and t-SNE) to produce point cloud charts for the interpretation of the obtained descriptors. In addition, a Bag of Visual Words was defined and used as input for image classification and content-based retrieval.

As part of our experiments, we've also applied the aforementioned analysis to an extended version of the dataset, including synthetic images created by data augmentation and background/foreground segmentantion methods. 

In order to draw conclusions and lessons learned from the project, we've compared the accuracy of the classification models developed for each combination of images and features, and also analyzed the results of the image retrieval system.

The inputs were obtained by taking photos of the Scrap / Waste Disposal operation of a large IT company, which allowed us to get a variety of electronic waste images, like Printed Circuit Boards, Connectors, Cables and Wires, Batteries, Screens, Magnetic Tapes, Metals, etc.

# 1. Main Objective
Analyze a set of images related to Electronic Waste by extracting texture and keypoint features and then using these features in two tasks: image classification and image retrieval.

# 2. Description of the Input Images
As shown in the examples below, the images refer to Electronic Waste (Printed Circuit Boards, Connectors, Cables and Wires, Batteries, Screens, Magnetic Tapes, Metals, etc.) and they were obtained by taking photos of the Scrap / Waste Disposal operation of a large IT company. Given the main objective of the project and the methods chosen, we don't propose the production of output images. On the other hand, we apply dimension reduction techniques and plot point cloud charts to create a visualization of the features extracted from the images.

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

GLCM is a statistical method of examining texture that considers the spatial relationship of pixels in the Gray-Level Co-Occurrence matrix, also known as the gray-level spatial dependence matrix. The GLCM functions characterize the texture of an image by calculating how often pairs of pixel with specific values and in a specified spatial relationship occur in an image, creating a GLCM, and then extracting statistical measures from this matrix. We used haralick feature extraction implemented in the Mahotas package for Python. Mahota's implementation contemplates the first 13 features from Haralick's original work. They are: (1) Angular Second Moment, (2) Contrast, (3) Correlation, (4) Variance, (5) Inverse Difference Moment, (6) Sum Average, (7) Sum Variance, (8) Sum (9) Entropy, (10) Difference Variance, (11) Difference Entropy and (12 and 13) Information Measures of Correlation.

LBP is a texture descriptor that takes the surrounding points of a central pixel and tests whether they are greater than or less than the central point (i.e. gives a binary result). By doing so it creates both uniform and non-uniform patterns indicating changes or regions of similar textures. These patterns then have their distribuitions analysed via their histogram. We used LBP implementation available in the scikit-image package for Python.

BRISK is a feature point detection and description algorithm with scale and rotation invariance. It constructs the feature descriptor of the local image through the gray scale relationship of random point pairs in the neighborhood of the local image, and obtains the binary feature descriptor. BRISK was proposed by S. Leutenegger, M. Chli and R. Y. Siegwart in 2011 as a novel method for keypoint detection, description and matching, delivering high quality performance as in state-of-the-art algorithms, such as SIFT and SURF, albeit at a dramatically lower computational cost.

After applying these feature extraction techniques, we used the k-means algorithm to create clusters for the descriptors. We tried a range of values from 25 to 1,000 for the number of clusters and decided the best option by comparing their Silhouette Scores. Then we applied two Multidimension Projection methods to reduce the high cardinallity of the data in order to plot 2D point cloud charts of the two main components obtained from PCA and t-SNE.

Results are discussed in detail in the next section, but clustering and visualizing the point clouds of the main components didn't produce overall good outputs. So we decided to use the extracted features as inputs in Machine Learning classification algorithms for better understanding and interpretation. So we used six algorithms (namely Logistic Regression, Multi-Layer Perceptron, Support Vector Machine, Decision Tree, Random Forest and eXtreme Gradient Boosting) in a cross-validation training procedure with 5 Stratified Folds and measured the mean accuracy of each model to proceed with the one with the highest value.

We then trained the best model for each set of features (extracted by the three different methods) and used these models to predict the labels (or classes) of images from a a test dataset and, again, measured not only the overall accuracy of the models, but also the individual accuracy per class.

The whole procedure of extracting the features, clustering and visualizing their descriptors, as well as training machine learning models to predict the classes of new images, was replicated to 4 sets of images, as describred below:

1 - Imbalanced dataset, including the original images with background
2 - Imbalanced dataset, including the images after background segmentation
3 - Balanced dtaaset, including the original images with background
4 - Balanced dataset, including the images after background segmentation

The balanced dataset was obtained with the application of a data augmentation technique provided by the Keras package on TensorFlow platform. With the objective of balancing the number of elements in each class, we created new images by applying small variations, such as width and height changes, 90 degrees rotation, brigthness adjustments and horizontal flip.

As for the background/foreground segmentation, we initially tried to apply a Region Growing Segmentation technique, firstly getting the corners of the object in the image to be used as seeds, but it didn't produce the desired results. We've also tried to use an HSV threshold filter, applying the mask generated by the binary threholding, which couldn't completely isolte the background in the input inputs.

Then, as a support to the main objective of this project, which is the exploration of feature extraction techniques, we decided to use a U2-Net pre-trained Neural Network, which is a two-level nested Neural Network using an U-structure architecture that is designed for salient object detection (SOD). The architecture allows the network to go deeper, attain high resolution, without significantly increasing the memory and computation cost. This is achieved by a nested U-structure: on the bottom level, with a novel ReSidual U-block (RSU) module, which is able to extract intra-stage multi-scale features without degrading the feature map resolution; on the top level, there is a U-Net like structure, in which each stage is filled by a RSU block. The results were much better than the previous attempts.

Besides realying on machine learning classification algorithms to explore the extracted features, we also opted to build a content-based image retrieval system for each of the 4 sets of images described above, particularly using GLCM and LBP as feature descriptors. The visual words were obtained with k-means applied over patches of the input images, so that we could get the nearest images related to the query input.

# 4. Obtained Results and Discussion

To exemplify the outputs produced by the data augmentation and background/foreground segmentation applied in this project we present the set of imagens below, which shows the orinal version of a water pump on the top left hand side, the rotated water pump obtained by data augmentation on the top right hand side, the water pump without the image background segmented by the U2-Net on the bottom left hand side and the water pump rotated and segmented on the bottom right hand side.

<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train/Water%20Pump/Water%20Pump1.JPG" width="300"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train_balanced/Water%20Pump/Water%20Pump1_rotation-range_24.JPG" width="300">

<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train_segmented/Water%20Pump/Water%20Pump1_segmented.JPG" width="300"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train_segmented_balanced/Water%20Pump/Water%20Pump1_segmented_rotation-range_24.JPG" width="300">

Generally speaking, the use of the U2-Net Neural Network produced good object detection and highlighting. Results can be viewed at https://github.com/josealexandre-mecai/img-processing-2022/tree/main/waste_images/train_segmented.

#### Original image set         
|Model   | BRISK    | GLCM     | LBP      |
|--------|----------|----------|----------|
|Logistic|__0.3965__|0.3080    |0.3486    |
|MLP     |0.3310    |0.3027    |0.2556    |
|SVM     |0.3398    |__0.3115__|0.3047    |
|Dec Tree|0.3398    |0.2850    |0.2646    |
|Rand For|0.3929    |0.3062    |__0.3650__|
|XGBoost |0.3858    |0.2973    |0.3486    |

#### Balanced image set         
|Model   | BRISK    | GLCM     | LBP      |
|--------|----------|----------|----------|
|Logistic|0.2546    |0.1103    |0.1606    |
|MLP     |0.2626    |0.1183    |0.1255    |
|SVM     |0.1603    |0.1082    |0.1070    |
|Dec Tree|0.1804    |0.1063    |0.1502    |
|Rand For|__0.2826__|__0.1323__|__0.1852__|
|XGBoost |0.2547    |0.1162    |0.1461    |

#### Segmented image set         
|Model   | BRISK    | GLCM     | LBP      |
|--------|----------|----------|----------|
|Logistic|0.3681    |0.2761    |0.2793    |
|MLP     |0.3062    |0.2726    |0.2063    |
|SVM     |0.3115    |0.2779    |0.3047    |
|Dec Tree|0.3168    |__0.2850__|0.2007    |
|Rand For|__0.4000__|0.2779    |__0.3230__|
|XGBoost |0.3912    |0.2761    |0.2573    |

#### Balanced and Segmented image set         
|Model   | BRISK    | GLCM     | LBP      |
|--------|----------|----------|----------|
|Logistic|0.2426    |0.1062    |0.1550    |
|MLP     |0.2225    |0.1082    |0.1383    |    
|SVM     |0.1202    |0.1223    |0.1217    |    
|Dec Tree|0.1683    |0.1143    |0.0963    |    
|Rand For|__0.2946__|__0.1244__|0.1466    |    
|XGBoost |0.2425    |0.1163    |__0.1634__|    
   
# 5. Descriptions of the roles of the student in the project

José was resposible for...
Meanwhile, Gustavo was responsible for: the Clustering and Silhouette Score applications, the Multidimensional Projections and point clouds plotting, the cross-validation training procedure for the machine learning models and writing the final report.

# 6. Demo of the project

Please follow and run the following notebook:
https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Showcase%20Predictors.ipynb

# 7. Presentation

The video presentation of this project can be viewed at: YouTube link.
