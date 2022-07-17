import cv2
import numpy as np
import pandas as pd
import os
import joblib as jb

# Importing display from IPython package
from IPython.display import Image
#from PIL import Image

# Importing metrics tools
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Importing visualization tools
import seaborn as sn
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.colors as mcolors


# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

# Importing feature extraction tools
from skimage.feature import greycomatrix, greycoprops
import mahotas as mt # Mahotas library for GLCM calculation
from skimage import feature # feature.local_binary_pattern for LBP calculation

# BRISK - Feature Point Predictor
def brisk_predictor(image_path, option):
    '''
    Predict class based on BRISK features

    Parameters:
        img_path: path for an image
        option: dataset option

    Returns:
        predicted class, probabilities
    '''

    # Read the image
    im = cv2.imread(image_path)

    # Select the model based on the input option
    if option == '1':
        model = "bovw_brisk_lg.pkl"
    elif option == '2':
        model = "bovw_segmented_brisk_rf.pkl"
    elif option == '3':
        model = "bovw_balanced_brisk_rf.pkl"
    elif option == '4':
        model = "bovw_segmented_balanced_brisk_rf.pkl"
    else:
        print("Please type the correct option: 1, 2, 3 or 4")

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    # from stored pickle file (generated during training)
    clf, classes_names, stdSlr, k, voc = jb.load(model)

    # Extract features with BRISK method
    # Create List where all the descriptors will be stored
    des_list = []

    brisk = cv2.BRISK_create(30)
    kpts, des = brisk.detectAndCompute(im, None)
    des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Calculate the histogram of features
    # vq Assigns codes from a code book to observations.
    im_features = np.zeros((1, k), "float32")

    for i in range(0,1):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    im_features = stdSlr.transform(im_features)
    prediction = [classes_names[i] for i in clf.predict(im_features)]

    probabilities = {}
    for prob, class_name in zip(clf.predict_proba(im_features)[0], classes_names):
        probabilities[class_name] = round(prob, 4)

    return prediction, probabilities


def extract_glcm_features(image):
    '''
    Extract GLCM features

    Parameters:
        img: image file

    Returns:
        texture features
    '''

    # Calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)
    return textures


def glcm_predictor(image_path, option):
    '''
    Predict class based on GLCM features

    Parameters:
        img_path: path for an image
        option: dataset option

    Returns:
        predicted class, probabilities
    '''

    # Read the image
    im = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Select the model based on the input option
    if option == '1':
        model = "bovw_glcm_svm.pkl"
    elif option == '2':
        model = "bovw_segmented_glcm_dt.pkl"
    elif option == '3':
        model = "bovw_balanced_glcm_rf.pkl"
    elif option == '4':
        model = "bovw_segmented_balanced_glcm_rf.pkl"
    else:
        print("Please type the correct option: 1, 2, 3 or 4")

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    # from stored pickle file (generated during training)
    clf, classes_names, stdSlr, k, voc = jb.load(model)

    # Extract features with GLCM method
    # Create List where all the descriptors will be stored
    des = extract_glcm_features(gray_img)
    des_list = []
    des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Calculate the histogram of features
    # vq Assigns codes from a code book to observations.
    im_features = np.zeros((1, k), "float32")

    for i in range(0,1):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1

    im_features = stdSlr.transform(im_features)
    prediction = [classes_names[i] for i in clf.predict(im_features)]

    probabilities = {}
    for prob, class_name in zip(clf.predict_proba(im_features)[0], classes_names):
        probabilities[class_name] = round(prob, 4)

    return prediction, probabilities


def extract_lbp_features(img, radius=1, sampling_pixels=8):
    '''
    Extract LBP features

    Parameters:
        img: image file
        radius: radius of circle (spatial resolution of the operator)
        sampling_pixels: number of circularly symmetric neighbor set points (quantization of the angular space)

    Returns:
        texture features
    '''

    # LBP operates in single channel images so if RGB images are provided
    # we have to convert it to grayscale
    if (len(img.shape) > 2):
        img = img.astype(float)
        # RGB to grayscale convertion using Luminance
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11

    # converting to uint8 type for 256 graylevels
    img = img.astype(np.uint8)

    # normalize values can also help improving description
    i_min = np.min(img)
    i_max = np.max(img)
    if (i_max - i_min != 0):
        img = (img - i_min) / (i_max - i_min)

    # compute LBP
    lbp = feature.local_binary_pattern(img, sampling_pixels, radius, method="uniform")

    '''
    # LBP returns a matrix with the codes, so we compute the histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))

    # normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    # return the histogram of Local Binary Patterns
    return hist
    '''

    return lbp


def lbp_predictor(image_path, option):
    '''
    Predict class based on LBP features

    Parameters:
        image_path: path for an image
        option: dataset option

    Returns:
        predicted class, probabilities
    '''

    # Read the image
    im = cv2.imread(image_path)

    # Select the model based on the input option
    if option == '1':
        model = "bovw_lbp_rf.pkl"
    elif option == '2':
        model = "bovw_segmented_lbp_rf.pkl"
    elif option == '3':
        model = "bovw_balanced_lbp_rf.pkl"
    elif option == '4':
        model = "bovw_segmented_balanced_lbp_xgboost.pkl"
    else:
        print("Please type the correct option: 1, 2, 3 or 4")

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    # from stored pickle file (generated during training)
    clf, classes_names, stdSlr, k, voc = jb.load(model)

    # Extract features with LBP method
    # Create List where all the descriptors will be stored
    des = extract_lbp_features(im)
    des_list = []
    des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Workaround for shape[1] != 640
    des_list_adj = np.zeros((des_list[0][1].shape[0], 640))

    for i in range(des_list_adj.shape[0]):
        for j in range(des_list_adj.shape[1]):
            des_list_adj[i][j] = des_list[0][1][i][j]

    # Calculate the histogram of features
    # vq Assigns codes from a code book to observations.
    im_features = np.zeros((1, k), "float32")

    for i in range(0,1):
        words, distance = vq(des_list_adj,voc)
        for w in words:
            im_features[i][w] += 1

    im_features = stdSlr.transform(im_features)
    prediction = [classes_names[i] for i in clf.predict(im_features)]

    probabilities = {}
    for prob, class_name in zip(clf.predict_proba(im_features)[0], classes_names):
        probabilities[class_name] = round(prob, 4)

    return prediction, probabilities

def prob_graph(prob_brisk, prob_glcm, prob_lbp):
    '''
    Generate graph with class probabilities

    Parameters:
        prob_brisk: list of class probabilities obtained from BRISK prediction
        prob_glcm: list of class probabilities obtained from GLCM prediction
        prob_lbp: list of class probabilities obtained from LBP prediction

    Returns:
        graph with class probabilities
    '''

    names = list(prob_brisk.keys())

    values_list=[]
    values_list.append(list(prob_brisk.values()))
    values_list.append(list(prob_glcm.values()))
    values_list.append(list(prob_lbp.values()))
    values_list = [[100 * i for i in inner] for inner in values_list]

    fig, (axs) = plt.subplots(1, 3, figsize=(20, 4))

    for values, i in zip(values_list, range(len(values_list))):
        norm = plt.Normalize(0, max(values))
        colors = plt.cm.Purples(norm(values))

        if i==0:
            title = 'BRISK'
        elif i==1:
            title = 'GLCM'
        elif i==2:
            title = 'LBP'

        axs[i].bar(range(len(values)), values, tick_label=names, color=colors)
        axs[i].set_title(title + ' - Class Probabilities - %')
        axs[i].set_xticklabels(names, rotation=90, ha='right')
        axs[i].set_ylim(0, 100)

        for p in axs[i].patches:
            axs[i].annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.001, p.get_height() * 1.07), color='black')

    plt.show()


def predict(im_path, opt):
    '''
    Predict classes based on BRISK, LBP and GLCM features

    Parameters:
        im_path: path for an image
        option: dataset option

    Returns:
        prints predictions, probabilities and draw graph with class probabilities
    '''

    pred_brisk = brisk_predictor(im_path, opt)[0]
    prob_brisk = brisk_predictor(im_path, opt)[1]

    print("Prediction BRISK: ", pred_brisk, "\n")
    print("Probabilities BRISK:\n", prob_brisk, "\n")

    pred_glcm = glcm_predictor(im_path, opt)[0]
    prob_glcm = glcm_predictor(im_path, opt)[1]

    print("Prediction GLCM: ", pred_glcm, "\n")
    print("Probabilities GLCM: \n", prob_glcm, "\n")

    pred_lbp = lbp_predictor(im_path, opt)[0]
    prob_lbp = lbp_predictor(im_path, opt)[1]

    print("Prediction LBP: ", pred_lbp, "\n")
    print("Probabilities LBP: \n", prob_lbp, "\n")

    prob_graph(prob_brisk, prob_glcm, prob_lbp)

