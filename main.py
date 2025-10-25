import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

'''
    Variables that affect the below functions. Set accordingly!
'''
#   Variables for use in the program.
folds = 5       # Determines how many different folds occur. (High effect on performance!!!)
neighbors = 10  # Determines how many neighbors will be considered. (Low effect on performance.)


'''
    standardize()
        Helper function to standardize/normalize an array of data.
        Input an array.
        Returns an equally-sized array with standardized values.
'''
#   Helper function to standardize an array of data
def standardize(inputArr):
    mean = inputArr.mean(axis = 0)
    std = inputArr.std(axis = 0)
    std[std == 0] = 1
    return (inputArr - mean) / std

'''
    kfold()
        Helper function to divide an array into k equally sized parts.
        Input an array.
        Returns an array of k arrays, splitting the original array.
        (Useful for five-fold validation, a replacement to random validation sets.)
'''
#   Helper function to divide an array into k equal parts
def kfold(inputArr):
    height = inputArr.shape[0]
    returnArr = []
    for i in range(folds):
        returnArr.append(inputArr[(int) (height/folds * i) : (int) (height/folds * (i + 1))])
    return returnArr

'''
    plotimg()
        Helper function to print out a specified range of images and their associated keypoints.
        Input an array representation of an image, an array representation of keypoints, and a tuple containing the desired range.
        Returns nothing, but displays all images within that range in a new window.
        (Similar to Homework 1, useful for checking out what's going on.)
'''
#   Helper function to print out a specified range of images and their associated keypoints
def plotimg(imageArr, keypoints, indexrange):
    rangeDiff = indexrange[1] - indexrange[0]
    fig, ax = plt.subplots(1, rangeDiff, figsize=(5, 5))
    if (rangeDiff == 1):
        ax = [ax]
    for increment in range(rangeDiff):
        index = indexrange[0] + increment
        image = imageArr[index].reshape((96, 96))
        x_coords = keypoints[index][0::2]
        y_coords = keypoints[index][1::2]

        ax[increment].imshow(image, cmap=plt.get_cmap("gray"))
        ax[increment].set_title(index)
        ax[increment].scatter(x_coords, y_coords)
        increment += 1
    plt.show()


#   Loading data
print("Loading data...")
train_frame = pd.read_csv('data\\training.csv').dropna()
test_frame = pd.read_csv('data\\test.csv')

print("Reformatting data...")
#print(train_frame.head(1))
train_points = train_frame.values[:, : -1]
train_images_str = train_frame.values[:, -1]
#print(train_points.shape)
#print(train_image.shape)
#print(train_image.shape[0])
train_images = []
for i in range(train_images_str.shape[0]):
    train_images.append(np.array(train_images_str[i].split(), dtype = float))
train_images = np.array(train_images)
train_points = np.array(train_points.astype(float))

print("Printing images...")
plotimg(train_images, train_points, (0, 5))

print("Dividing data...")
train_points_folds = kfold(train_points)
train_image_folds = kfold(train_images)

print("Started training (k-fold validation).")
train_predictions = []
for iter in range(folds):
    print(f"Starting fold {iter + 1}/{folds}...")
    train_trainfolds_points = []
    train_trainfolds_images = []
    train_validfolds_points = []
    train_validfolds_images = []
    for i in range(folds):
        if (iter != i):
            train_trainfolds_images.append(train_image_folds[i])
            train_trainfolds_points.append(train_points_folds[i])
        else:
            train_validfolds_images = train_image_folds[i]
            train_validfolds_points = train_points_folds[i]
    train_trainfolds_images = np.vstack(train_trainfolds_images)
    train_trainfolds_points = np.vstack(train_trainfolds_points)

    #print(train_trainfolds_points.shape)
    #print(train_trainfolds_images.shape)
    #print(train_validfolds_points.shape)
    #print(train_validfolds_images.shape)

    for v in train_validfolds_images:
        distances = np.sum((train_trainfolds_images - v) ** 2, axis = 1)
        sortedindices = np.argsort(distances)
        nearestpoints = []
        for i in range(neighbors):
            nearestpoints.append(train_trainfolds_points[sortedindices[i]])
        nearestpoints = np.array(nearestpoints)
        avgpoints = nearestpoints.mean(axis = 0)
        train_predictions.append(avgpoints)
    

print("Training finished!")
train_predictions = np.array(train_predictions)

print("Metrics:")
#print(f"train_points.shape: {train_points.shape}")
#print(f"train_predictions.shape: {train_predictions.shape}")
print(f"MSE: {mean_squared_error(train_points, train_predictions)}")
print(f"MAE: {mean_absolute_error(train_points, train_predictions)}")