# digit_recognizer
This code was performed using the MNIST data available on the [Kaggle Platform](https://www.kaggle.com/competitions/digit-recognizer). A simple Artifical Neural Network was produced with one hidden layer to predict the number based on the pixel values. In addition to utilizing the data available from MNIST, random noise were added to the training dataset to explore how it affects the accuracy. 

___
### Hidden Layer Nodes
When creating the model, two configurations were tried 10 and 128 nodes. The latter provided better model accuracy on the evaluation set 91% with an average training time of 30mins while the former layer provided a 87% accuracy with an average training time of 8mins. Since time was not a constraint, the larger model was adopted. More complex models can be utilized to achieve higher accuracy however it can lead to overfitting.
___
### Augmenting Dataset with Adversarial Attack
Random noise were added to the training dataset to explore the effects of adversarial attacks on the model's prediction accuracy. The random noise was generated by creating an array (same size and input data) with random numbers following gaussian distribution of mean 0 and standard deviation of 0.1. The noise was then added to the intial input and all values were forced into the [0,1] pixel range. A noise parameter was added to control the magnitude to the noise. 
| Model Trained with Noise Parameter | Model Accuracy on Eval Set|
| ---------------------------------- | ------------------------- |
| 0 (Base Model)                     | 91.857%                   |
| 0.1                                | 93.095%                   |
| 0.2                                | 93.095%                   |
| 0.3                                | 93.095%                   |

The table shows that addition of noise to augment the data does help to boost the performance of the model however it saturated at the 0.1 noise parameter. Further exploration can be computed similar to how hyperparameters are optimized. 

#### MNIST Sample with No Noise
![no noise](https://github.com/RobinRamdin13/digit_recognizer/blob/main/plots/0.0_samples.jpeg)
#### MNIS Sample with 0.1 Noise
![0.1 noise](https://github.com/RobinRamdin13/digit_recognizer/blob/main/plots/0.1_samples.jpeg)
#### MNIS Sample with 0.2 Noise
![0.2 noise](https://github.com/RobinRamdin13/digit_recognizer/blob/main/plots/0.2_samples.jpeg)
#### MNIS Sample with 0.3 Noise
![0.3 noise](https://github.com/RobinRamdin13/digit_recognizer/blob/main/plots/0.3_samples.jpeg)

___
### Running the Code 
#### Creating Virtual Environment
To create the virtual environment run the following code in your terminal, you can rename `env` to any name you want for your virtual environment.`python -m venv env`.

In the event the virtual environment has not yet been activate, you need to run the following command: `env\Scripts\activate.bat`. This might defer based on which machine you're using, I was using Visual Studio Code on a Windows and the command prompt as terminal. 

#### Install all the dependencies 
After creating the virtual environment, run the following command, this will download all the required libraries required to replicate the code. `python pip install -r requirements.txt`

#### Executing main.py
To run the main file run the following command within your terminal `python main.py`.

___
### Comments and Contribution 
This is a fun project that I took from kaggle, any comment or contribution are welcome.