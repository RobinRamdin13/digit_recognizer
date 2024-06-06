import os 
import numpy as np 
import pandas as pd 

from tqdm import tqdm 
from typing import List, Tuple
from sklearn.model_selection import train_test_split

# set the random seed for reproducibility
np.random.seed(12345)
random_state = 12345

def add_noise(mu, sigma, x, y, noise_param):
    gaussian_dist = np.random.normal(mu, sigma, size=x.shape)
    x_noise = np.maximum(np.maximum(gaussian_dist,0)*noise_param + x, 1.0)
    x = np.append(x, x_noise, axis=1)
    y = np.append(y, y, axis=0)
    return x, y

def init_params()-> Tuple[List[float], List[float], List[float], List[float]]:
    """Function to initials weights and biases with random variables between -0.5 and 0.5

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: weights and biases for neural network layers
    """
    w1 = np.random.rand(128, 784) - 0.5 # weights of first layer 
    b1 = np.random.rand(128, 1) - 0.5 # bias for first layer 
    w2 = np.random.rand(10, 128) - 0.5 # weights for second layer 
    b2 = np.random.rand(10, 1) - 0.5 # bias for second layer
    return w1, b1, w2, b2

def relu(z: List[float]) -> List[float]:    
    """Function to perform rectified linear unit on data

    Returns:
        List[float]: array with negative terms converted to zero
    """    
    z = np.maximum(z, 0)
    return z

def relu_der(z: List[float])-> List[float]:
    """Function to compute the derivative of relu

    Args:
        z (List[float]): array of linear transformation from first layer

    Returns:
        List[float]: array of values greater than zero
    """    
    return z>0

def softmax(z: List[float])-> List[float]: 
    """Function to compute the softmax of an array

    Args:
        z (List[float]): array of output from hidden layer

    Returns:
        List[float]: array of softmax
    """    
    z = np.exp(z) / sum(np.exp(z))
    return z

def forward_prop(w1: List[float], b1:List[float], w2:List[float], b2:List[float], x:List[float])-> Tuple[List[float], 
                                                                                                         List[float], 
                                                                                                         List[float], 
                                                                                                         List[float]]:
    """Function to compute the forward propagation

    Args:
        w1 (List[float]): weights for first layer
        b1 (List[float]): bias for first layer
        w2 (List[float]): weights for second layer
        b2 (List[float]): bias for second layer
        x (List[float]): mnist pixel data

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: Arrays containing the respective output of each layer
    """    
    z1 = w1.dot(x) + b1 # linear transformation of first layer
    a1 = relu(z1) # non-linear transformation
    z2 = w2.dot(a1) + b2 # linear transformation of second layer
    a2 = softmax(z2) # non-linear transformation
    return z1, a1, z2, a2

def one_hot_encode(y:List[int])-> List[int]:
    """Function to one hot encode the categorical values

    Args:
        y (List[int]): array of initial categorical values

    Returns:
        List[int]: array of one hot encoded caategorical values
    """    
    one_hot = np.zeros((y.size, y.max() + 1))  # create empty array
    one_hot[np.arange(y.size), y] = 1 # perform one hot encoding
    one_hot = one_hot.T # perform the transpose
    return one_hot

def back_prop(z1:List[float], a1:List[float], z2:List[float], a2:List[float], 
              w1:List[float], w2:List[float], x:List[float], y:List[int], m:int)-> Tuple[List[float], 
                                                                                         List[float],
                                                                                         List[float],
                                                                                         List[float]]:
    """Function to compute the back propagation

    Args:
        z1 (List[float]): array of linear transformation of first layer
        a1 (List[float]): array of non linear tranformation of first layer
        z2 (List[float]): array of linear transformation of second layer
        a2 (List[float]): array of non-linear transformation of second layer
        w1 (List[float]): array of weights for first layer
        w2 (List[float]): array of weights for second layer
        x (List[float]): mnist pixel data
        y (List[int]): categorical values
        m (int): number of pixels

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: Array containing the respective derivative of weights and bias
    """    
    one_hot_Y = one_hot_encode(y)
    dz2 = a2 - one_hot_Y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * relu_der(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2

def param_update(w1:List[float], b1:List[float], w2:List[float], b2:List[float], 
                 dw1:List[float], db1:List[float], dw2:List[float], db2:List[float], alpha:float)->Tuple[List[float],
                                                                                                         List[float],
                                                                                                         List[float],
                                                                                                         List[float]]:
    """Function to update the weights and bias of each layer

    Args:
        w1 (List[float]): weights for first layer
        b1 (List[float]): bias for first layer
        w2 (List[float]): weights for second layer
        b2 (List[float]): bias for second layer
        dw1 (List[float]): weight derivative for first layer
        db1 (List[float]): bias derivative for first layer
        dw2 (List[float]): weight derivative for second layer
        db2 (List[float]): bias derivative for second layer
        alpha (float): learning rate

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: updated weights and bias for each layer 
    """    
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def prediction(a2:List[float])->List[float]:
    """Function to compute the category prediction based on softmax data

    Args:
        a2 (List[float]): softmax output from model 

    Returns:
        List[float]: category prediction
    """    
    return np.argmax(a2, 0)

def accuracy(y_pred:List[int], y:List[int])->float:
    """Function to compute the accuracy of predictions

    Args:
        y_pred (List[int]): predicted values
        y (List[int]): true values

    Returns:
        float: accuracy of prediction
    """    
    print(f"y_pred values : {y_pred}, y_true values {y}")
    return np.sum(y_pred == y) / y.size
    
def gradient_descent(x:List[float], y:List[int], m:int, alpha:float, iter:int)->Tuple[List[float],
                                                                                      List[float], 
                                                                                      List[float],
                                                                                      List[float]]:
    """Function to perform gradient descent

    Args:
        x (List[float]): mnist pixel data
        y (List[int]): categorical values
        m (int): number of pixels
        alpha (float): learning rate
        iter (int): number of training iterations

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: final weights and bias of each layer after training
    """    
    w1, b1, w2, b2 = init_params()
    for i in tqdm(range(iter), desc='Training Model'):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, x, y, m)
        w1, b1, w2, b2 = param_update(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        # if i % 1000 == 0:
        #     print(f'Training Iteration: {i}')
        #     print(f"Training Prediction Accuracy: {accuracy(prediction(a2), y)}")
    return w1, b1, w2, b2

def main(train_path:str, test_path:str, output_path:str)-> None:
    # load the training and test data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # # split training into training and evaluation set 
    y = df_train.pop('label').to_frame()
    X = df_train
    
    # use stratify to ensure balance label split  
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)
    m, _ = np.array(X_train).shape

    # convert to arrays and reshape with transpose 
    X_train, X_eval, y_train, y_eval = np.array(X_train).T, np.array(X_eval).T, np.ravel(np.array(y_train)), np.ravel(np.array(y_eval))
    X_train, y_train = add_noise(mu=0, sigma=0.1, x=X_train, y=y_train, noise_param=0.3)
    
    # normalize the X values by dividing by 255 
    X_train, X_eval = X_train/255, X_eval/255

    # train model 
    w1, b1, w2, b2 = gradient_descent(X_train, y_train, m, alpha=0.01, iter=10000)

    # perform inference on eval set
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, X_eval)
    print(f"Evaluation Prediction Accuracy: {accuracy(prediction(a2), y_eval)}")
    return 

if __name__ == '__main__':
    cwd = os.getcwd()
    train_path = os.path.join(cwd, 'data/train.csv')
    test_path = os.path.join(cwd, 'data/test.csv')
    output_path = ''
    main(train_path, test_path, output_path)