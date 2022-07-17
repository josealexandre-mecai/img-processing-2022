import cv2
import numpy as np
import os
from os import listdir # to list files in a folder
import joblib as jb
import glob
import mahotas as mt

import matplotlib.pyplot as plt

import imageio
from imageio import imread

# Importing display from IPython package
from IPython.display import Image

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed # to execute function in parallel
from skimage import feature


def get_patches(img_file, random_state, patch_size=(11, 11), n_patches=250):
    '''
    Extracts subimages

    Parameters:
        img_file: path for an image
        patch_size: size of each patch
        n_patches: number of patches to be extracted

    Returns:
        images patches
    '''

    img = imread(img_file)

    # Extract subimages
    patch = extract_patches_2d(img,
                               patch_size=patch_size,
                               max_patches=n_patches,
                               random_state=random_state)

    return patch.reshape((n_patches,
                          np.prod(patch_size) * len(img.shape)))


def glcm_features(img, sampling_pixels=8):
    '''
    Extract GLCM features

    Parameters:
        img: image file
        sampling_pixels: number of circularly symmetric neighbor set points (quantization of the angular space).

    Returns:
        texture features
    '''

    # Converting to grayscale
    if (len(img.shape) > 2):
        img = img.astype(float)
        # RGB to grayscale convertion using Luminance
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11

    # Converting to uint8 type for 256 graylevels
    img = img.astype(np.uint8)

    '''
    # Normalize values can also help improving description
    i_min = np.min(img)
    i_max = np.max(img)
    if (i_max - i_min != 0):
        img = (img - i_min)/(i_max-i_min)
    '''

    # Calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(img)

    # Computing the histogram of features
    (hist, _) = np.histogram(textures.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))

    # Normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Return the histogram of features
    return hist


def lbp_features(img, radius=1, sampling_pixels=8):
    '''
    Extract LBP features

    Parameters:
        img: image file
        radius: radius of circle (spatial resolution of the operator)
        sampling_pixels: number of circularly symmetric neighbor set points (quantization of the angular space).

    Returns:
        texture features
    '''

    # LBP operates in single channel images so if RGB images are provided
    # we have to convert it to grayscale
    if (len(img.shape) > 2):
        img = img.astype(float)
        # RGB to grayscale convertion using Luminance
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.11

    # Converting to uint8 type for 256 graylevels
    img = img.astype(np.uint8)

    # Normalize values can also help improving description
    i_min = np.min(img)
    i_max = np.max(img)
    if (i_max - i_min != 0):
        img = (img - i_min) / (i_max - i_min)

    # Compute LBP
    lbp = feature.local_binary_pattern(img, sampling_pixels, radius, method="uniform")

    # LBP returns a matrix with the codes, so we compute the histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))

    # Normalization
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Return the histogram of Local Binary Patterns
    return hist


def retrieve_images(dataset_path, query_path, option, feature_extractor):
    '''
    Retrieve content-based images

    Parameters:
        dataset_path: path to the dataset of images
        query_path: path to the query image
        option: dataset option
        feature_extractor: glcm or lbp

    Returns:
        graph with nearest images related to the query
    '''

    # Defining list of images from dataset_path
    imgs_path = []

    for filename in listdir(dataset_path):
        imgs_path.append(dataset_path+filename)

    # Get dataset patches
    # BOF parameters
    tam_patch = (15, 15)
    n_patches = 250
    random_state = 1

    # Total of images
    n_imgs = len(imgs_path)

    # Extract patches in parallel
    # returns a list of the same size of the number of images
    patch_arr = Parallel(n_jobs=-1)(delayed(get_patches)(arq_img,
                                                         random_state,
                                                         tam_patch,
                                                         n_patches)
                                    for arq_img in imgs_path)

    patch_arr = np.array(patch_arr, copy=True)
    patch_arr = patch_arr.reshape((patch_arr.shape[0] * patch_arr.shape[1],
                                   tam_patch[0], tam_patch[0], 3))

    # Get query patches
    query_patches = get_patches(query_path, random_state, tam_patch, n_patches)
    query_patches = np.array(query_patches, copy=False)

    query_patches = query_patches.reshape((query_patches.shape[0],
                                           tam_patch[0], tam_patch[0], 3))

    # print('Extracted patches')
    # print(query_patches.shape)

    # Select the model based on the input option and the feature extractor
    if option == '1' and feature_extractor == 'glcm':
        model = "kmeans_glcm_imbalanced.pkl"
    elif option == '1' and feature_extractor == 'lbp':
        model = "kmeans_lbp_imbalanced.pkl"
    elif option == '2' and feature_extractor == 'glcm':
        model = "kmeans_glcm_segmented_imbalanced.pkl"
    elif option == '2' and feature_extractor == 'lbp':
        model = "kmeans_lbp_segmented_imbalanced.pkl"
    elif option == '3' and feature_extractor == 'glcm':
        model = "kmeans_glcm_balanced.pkl"
    elif option == '3' and feature_extractor == 'lbp':
        model = "kmeans_lbp_balanced.pkl"
    elif option == '4' and feature_extractor == 'glcm':
        model = "kmeans_glcm_segmented_balanced.pkl"
    elif option == '4' and feature_extractor == 'lbp':
        model = "kmeans_lbp_segmented_balanced.pkl"
    else:
        print("Please type the correct option: 1, 2, 3 or 4 and/or the correct feature extractor: glcm or lbp")

    # Load the model and the dictionary size
    # from stored pickle file (generated during training)
    kmeans_model, n_dic = jb.load(model)

    # Obtaining glcm or lbp features for each patch
    patch_feat = []

    for pat in patch_arr:
        if feature_extractor == 'glcm':
            glcm = glcm_features(pat, 8)
            patch_feat.append(glcm)
        elif feature_extractor == 'lbp':
            lbp = lbp_features(pat, 2, 8)
            patch_feat.append(lbp)

    # Compute features for each image
    img_feats = []

    for i in range(n_imgs):
        # Predicting n_patches of an image
        y = kmeans_model.predict(patch_feat[i * n_patches: (i * n_patches) + n_patches])

        # Computes histogram and append in the array
        hist_bof, _ = np.histogram(y, bins=range(n_dic + 1), density=True)
        img_feats.append(hist_bof)

    img_feats = np.array(img_feats, copy=False)

    # Get GLCM and LBP features for extracted patches from query image
    query = []

    if feature_extractor == 'glcm':
        for pat in query_patches:
            glcm = glcm_features(pat, 8)
            query.append(glcm)

        suptitle = 'GLCM'

    elif feature_extractor == 'lbp':
        for pat in query_patches:
            lbp = lbp_features(pat, 2, 8)
            query.append(lbp)

        suptitle = 'LBP'

    query = np.array(query, copy=False)

    # Get visual words for query and computes descriptor
    y = kmeans_model.predict(query)
    query_feats, _ = np.histogram(y, bins=range(n_dic + 1), density=True)

    # Computes distance
    dists = []
    for i in range(n_imgs):
        diq = np.sqrt(np.sum((img_feats[i] - query_feats) ** 2))
        dists.append(diq)

    # Check the nearest images
    k = 8
    k_cbir = np.argsort(dists)[:k]

    imgq = imageio.imread(query_path)

    # Show image retrieval based on query
    plt.figure(figsize=(12, 8))
    plt.subplot(331);
    plt.imshow(imgq)
    plt.title('Query');
    plt.axis('off')

    imgs = []
    for i in range(k):
        imgs.append(imageio.imread(imgs_path[k_cbir[i]]))

        plt.subplot(3, 3, i + 2);
        plt.imshow(imgs[i])
        plt.suptitle(suptitle)
        plt.title('%d, %.4f' % (i, dists[k_cbir[i]]))
        plt.axis('off')
