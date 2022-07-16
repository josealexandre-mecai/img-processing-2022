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

<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train/Water%20Pump/Water%20Pump1.JPG" width="200"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train_balanced/Water%20Pump/Water%20Pump1_rotation-range_24.JPG" width="200">
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train_segmented/Water%20Pump/Water%20Pump1_segmented.JPG" width="200"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/train_segmented_balanced/Water%20Pump/Water%20Pump1_segmented_rotation-range_24.JPG" width="200">

Generally speaking, the use of the U2-Net Neural Network produced good object detection and highlighting. Results can be viewed at https://github.com/josealexandre-mecai/img-processing-2022/tree/main/waste_images/train_segmented.

### Clustering and Multidimensional Projections Visualizations

As mentioned in the previous section, we tested various sizes of clusters in a range from 25 to 1,000. Since many of the extracted descriptors generated hundreds of thousands of data, we decided to apply a random sample from each dataset and then proceed with the clustering in order to facilitate the execution of this step. All combinations of feature extraction methods and datasets presented better results for k=25 clusters. The Silhoutte Scores are displayed in the table below.

#### Silhouette Scores for k=25
|      |Original  |Balanced  |Segmented |Balanc+Seg|
|------|----------|----------|----------|----------|
|BRISK |0.30      |0.30      |0.30      |0.30      |
|GLCM  |0.49      |0.54      |0.52      |0.55      |
|LBP   |-0.02     |-0.02     |0.24      |0.25      |

Note that GLCM produced the best cluster segmentations, since the closer to one the silhouette score gets, the better. Also, balancing and highlighting the objects in the images improved the cluster segmentation for GLCM descriptors. Brisk's descriptors cluster segmentation showed to be indifferent to the image set. As for the LBP descriptors, the background/foreground segmentation caused great impact in the cluster segmentation.

The following set of images show 2D point cloud charts for the two main components obtained from PCA applied to the descriptors obtained from the three methods and images sets.

#### PCA two components point cloud for BRISK descriptors applied to the original, balanced, segmented and balanced-segmented image sets, respectively
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_BRISK_normal-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_BRISK_balanced-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_BRISK_segmented-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_BRISK_balanced-segmented-data.png" width="250">

The charts above show a coarse segmentation, as a reflection of the low Silhouette Scores and, also from the fact that the two main components explain only around 16% of the variance for all the image sets.

#### PCA two components point cloud for GLCM descriptors applied to the original, balanced, segmented and balanced-segmented image sets, respectively
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_GLCM_normal-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_GLCM_balanced-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_GLCM_segmented-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_GLCM_balanced-segmented-data.png" width="250">

GLCM got better cluster segregation (Silhouette Score near 0.5) and presented an interesting pattern for the descriptors data.

#### PCA two components point cloud for LBP descriptors applied to the original, balanced, segmented and balanced-segmented image sets, respectively
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_LBP_normal-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_LBP_balanced-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_LBP_segmented-data.png" width="250"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/PCA_LBP_balanced-segmented-data.png" width="250">

The pattern of the LBP descriptors data changed after the object highlighting procedure.

#### t-SNE two components point cloud for BRISK descriptors applied to the original, balanced, segmented and balanced-segmented image sets, respectively
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_BRISK_normal-data.png" width="230"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_BRISK_balanced-data.png" width="230"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_BRISK_segmented-data.png" width="275"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_BRISK_balanced-segmented-data.png" width="275">

The two components extracted by the t-SNE method didn't provide good visualization of the BRISJ descriptors' clusters.

#### t-SNE two components point cloud for GLCM descriptors applied to the original, balanced, segmented and balanced-segmented image sets, respectively
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_GLCM_normal-data.png" width="230"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_GLCM_balanced-data.png" width="230"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_GLCM_segmented-data.png" width="275"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_GLCM_balanced-segmented-data.png" width="275">

The GLCM method produced less descriptors to the segmented pictures, that's why we see relatively few points in the chart above. The curious shape also reveals fairly well segmented clusters for all the image sets.

#### t-SNE two components point cloud for LBP descriptors applied to the original, balanced, segmented and balanced-segmented image sets, respectively
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_LBP_normal-data.png" width="230"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_LBP_balanced-data.png" width="230"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_LBP_segmented-data.png" width="275"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/t-SNE_LBP_balanced-segmented-data.png" width="275">

The two components extracted by the t-SNE method show a clear change the in the data pattern after applying the object hightlighting procedure.

### Image Features used as input in Machine Learning - Training Results

The four tables below present the mean accuracy results of the 5-fold cross-validation training. The best model for each feature extraction method is highlighted in __bold__. Random Forest is the algorithm that most frequently produced better results, with 8 models being selected. We also have one Logistic Regression, one Support Vector Machines, one Decision Tree and one eXtreme Gradient Boosting. These selected models were used to train the datasets and then were used to predict the classes of a new set of un-seen images.

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

### Image Features used as input in Machine Learning - Training Results

Next we present the results of the overall accuracy for the predictions made by the models and highglight in __bold__ the best results for each image set. The Random Forest model fed by the features extracted by the LBP got the best overall accuracy for the Original image set. BRISK features got the best overall accuracy for the Balanced, Segmented and Balanced/Segmented image sets, using the Random Forest model for them all.

BRISK is the method that showed overall best accuracy results considering all the image sets. It also was the less affected by the reduction in the accuracy caused by the class balancing. Segmenting the object from the background produced great increase in the GLCM accuracy.

#### Predictions accuracy in the test dataset
|      |Original  |Balanced  |Segmented |Balanc+Seg|
|------|----------|----------|----------|----------|
|BRISK |0.38      |__0.27__  |__0.40__  |__0.30__  |
|GLCM  |0.28      |0.16      |0.37      |0.11      |
|LBP   |__0.40__  |0.21      |0.36      |0.20      |

Now we present the Confusion Matrixes obtained from the predictions using the machine learing models for each combination of feature extraction method and image set.

#### BRISK features Confusion Matrix for the Original and Balanced image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_BRISK_original-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_BRISK_balanced-data.png" width="400">

#### BRISK features Confusion Matrix for the Segmented and Balanced-Segmented image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_BRISK_segmented-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_BRISK_balanced-segmented-data.png" width="400">

#### GLCM features Confusion Matrix for the Original and Balanced image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_GLCM_original-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_GLCM_balanced-data.png" width="400">

#### GLCM features Confusion Matrix for the Segmented and Balanced-Segmented image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_GLCM_segmented-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_GLCM_balanced-segmented-data.png" width="400">

#### LBP features Confusion Matrix for the Original and Balanced image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_LBP_original-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_LBP_balanced-data.png" width="400">

#### LBP features Confusion Matrix for the Segmented and Balanced-Segmented image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_LBP_segmented-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/CM_LBP_balanced-segmented-data.png" width="400">

We also present barplots showing the accuracy for each class individually.

#### BRISK features individual accuracy for the Original and Balanced image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_BRISK_original-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_BRISK_balanced-data.png" width="400">

#### BRISK features individual accuracy for the Segmented and Balanced-Segmented image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_BRISK_segmented-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_BRISK_balanced-segmented-data.png" width="400">

#### GLCM features individual accuracy for the Original and Balanced image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_GLCM_original-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_GLCM_balanced-data.png" width="400">

#### GLCM features individual accuracy for the Segmented and Balanced-Segmented image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_GLCM_segmented-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_GLCM_balanced-segmented-data.png" width="400">

#### LBP features individual accuracy for the Original and Balanced image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_LBP_original-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_LBP_balanced-data.png" width="400">

#### LBP features individual accuracy for the Segmented and Balanced-Segmented image sets.
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_LBP_segmented-data.png" width="400"><img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Final%20Report%20Images/class_acc_LBP_segmented-data.png" width="400">



# 5. Descriptions of the roles of the student in the project

José was resposible for...
Meanwhile, Gustavo was responsible for: the Clustering and Silhouette Score applications, the Multidimensional Projections and point clouds plotting, the cross-validation training procedure for the machine learning models and writing the final report.

# 6. Demo of the project

Please follow and run the following notebook:
https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Showcase%20Predictors.ipynb

# 7. Presentation

The video presentation of this project can be viewed at: YouTube link.
