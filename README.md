#### Gustavo Contini Torres - Nro USP 5373002
#### Jose Alexandre F Silva - Nro USP 11908649
#
# Project Title: Feature Extraction Analysis on Electronic Waste
#
# Abstract: 
The purpose of our project is to analyze a set of images related to Electronic Waste, by working with Feature Extraction techniques such as GLCM and LBP, and then interpreting the results based on Distance Functions (Euclidean, Cosine, Manhattan) and Multidimensional Projection (t-SNE, MDS or PCA). In addition, with the application of a clustering algorithm, we will define a Bag of Visual Words for image classification and content-based image retrieval in order to identify the best combination of features.

The inputs were obtained by taking photos of the Scrap / Waste Disposal operation of a large IT company, which allowed us to get a variety of electronic waste images, like Printed Circuit Boards, Connectors, Cables and Wires, Batteries, Screens, Magnetic Tapes, Metals, etc.

# 1. Main Objective: 
Analyze a set of images related to Electronic Waste by extracting Texture Features and then interpreting the results to evaluate the best combinations.

# 2. Description of the Input Images:
As shown in the examples below, the images refer to Electronic Waste (Printed Circuit Boards, Connectors, Cables and Wires, Batteries, Screens, Magnetic Tapes, Metals, etc.) and they were obtained by taking photos of the Scrap / Waste Disposal operation of a large IT company. 

# 3. Description of the Steps to reach the Objective:
In order to extract the features, we will work with Gray-Level Co-Occurrence Matrix (GLCM) and LBP (Local Binary Patterns). 

GLCM is a statistical method of examining texture that considers the spatial relationship of pixels in the Gray-Level Co-Occurrence matrix, also known as the gray-level spatial dependence matrix. The GLCM functions characterize the texture of an image by calculating how often pairs of pixel with specific values and in a specified spatial relationship occur in an image, creating a GLCM, and then extracting statistical measures from this matrix.

LBP is a texture descriptor used for the property of high discrimination power. LBP labels each pixel in an image by comparing the gray level with the neighboring pixels and then assigning a binary number. A value of unity is assigned to the neighbors with gray level greater than the center pixel in a predefined patch, otherwise a value of zero. A binary number is then obtained and assigned to the center pixel.

For the analysis and interpretation of the results, we will apply Distance Functions (Euclidean, Cosine, Manhattan) and Multidimensional Projection (t-SNE, MDS or PCA). Also, the idea is to define a Bag of Visual Words for content-based image retrieval in order to identify the best combination of features.

# 4. Initital Code (testing basic functions): 
https://github.com/josealexandre-mecai/img-processing-2022/blob/main/Feature%20Extraction%20Analysis%20on%20Electronic%20Waste.ipynb


# Images - Examples:

## Battery
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/Battery100.JPG" width="300">

## Cable
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/Cable%20and%20Wire41.JPG" width="300">

## Connector
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/Connector4.JPG" width="300">

## Magnetic Tape
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/Magnetic%20Tape8.JPG" width="300">

## Printed Circuit Board
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/Printed%20Circuit%20Board139.JPG" width="300">

## Screen
<img src="https://github.com/josealexandre-mecai/img-processing-2022/blob/main/waste_images/Tube%20and%20Screen33.JPG" width="300">

