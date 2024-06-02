
## We'll be working with the famous titanic dataset available from the seaborn library.
## We'll build a multi-layer deep neural network using the Functional API to predict our target variable, survived, using other variables as predictors.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


## EDA:

df = sns.load_dataset('titanic')

#### checking dataset columns
print(df.columns)

#### checking the dimensions of the dataset
print(df.shape) # 891 rows and 15 columns

####checking the first 5 rows of the dataset
print(df.head(5))

#### checking for not-null value counts and dtypes of each column
print(df.info()) # null values in age, deck, embarked, and embark_town columns

#### checking the summary stats for each column
print(df.describe())

#### checking the distribution of target variable
print(df['survived'].value_counts())

#### visualizing the number of passengers that survived the sinking of the Titanic.
sns.countplot(data=df, x='survived')
plt.title('Distribution of Passenger Survival')
plt.xlabel('Survival (0=No, 1=Yes)')
plt.ylabel('Frequency')
plt.show() # more people died than survived.

#### visualizing the count of each category in the embark_town variable
sns.countplot(data=df, x='embark_town', hue='survived')
plt.title('Survival Count by Embarked Town')
plt.xlabel('Embark Town')
plt.ylabel('Frequency')
plt.show()

#### checking relationships between predictor and target variables:
#### Let's look at the distribution of gender with respect to survival
sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show() # shows that women had a higher probability of surviving than men.

#### Examining pclass and test the assumption that passenger class had an influence on the survival rate.
sns.countplot(data=df, x='pclass', hue='survived')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Frequency')
plt.show() # shows that the first class passengers had a higher chance of survival compared to the third class passengers.

## Data cleaning and Preprocessing:

### removing duplicate variables:
#### during the EDA we noticed that embark_town and embarked had same number of missing values, lets check further:
print(df['embark_town'].value_counts())
print(df['embarked'].value_counts()) # The output clearly shows that the variables embarked_town and embarked are duplicates. We can remove one of them.

print(df['survived'].value_counts())
print(df['alive'].value_counts()) # The output clearly shows that the variables survived and alive are duplicates. We can remove one of them.

print(df['pclass'].value_counts())
print(df['class'].value_counts()) # The output clearly shows that the variables pclass and class are duplicates. We can remove one of them.

### missing values treatment:
#### getting % of missing values in each column
percent_missing = (df.isnull().sum() / len(df)) * 100
print(percent_missing)
#### 19.9% in age, We'll impute these missing values with the median.
#### 77.2% in deck, This is a lot of missing values to reliably impute or to replace with a measure of central tendency; we'll remove this variable from our dataset.
#### 0.2% in embark_town, we'll impute the missing values for embark_town using the mode of this categorical variable. Looking at the output from df['embark_town'].value_counts() above, 'Southampton' is the mode.

df['embark_town'].fillna('Southampton', inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)
df = df.drop(['embarked', 'alive', 'class', 'deck'], axis=1)

### Dummy Encoding:
df = pd.get_dummies(df, columns=['embark_town','sex','adult_male','alone', 'who', 'pclass'], drop_first=True, prefix=['embark_town','sex','adult_male','alone', 'who', 'pclass'])

### Data Split
y = df['survived'] # or target_variable = ['survived'], then y=df[target_variable].values
X = df.drop('survived', axis=1) # or predictors = list(set(list(df.columns))-set(target_variable)), then X = df[predictors].values

### Normalization
#### Since the scale of the predictor variables varies, it's a good idea to bring them into a uniform scale. One such technique is called normalization, which scales the predictor variables to have values between 0 and 1.
X = X / X.max() # or df[predictors] = df[predictors] / df[predictors].max()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

print(X_train.shape); print(X_test.shape)


# Steps to define the deep neural network model with Functional API in TensorFlow:

import tensorflow as tf

## Define Input Layer:
#### Unlike the Sequential model, where the input shape is defined in the first hidden layer itself, the Functional model defines a standalone input layer that specifies the shape of input data.

input_layer = tf.keras.Input(shape=(X_train.shape[1],))

## Add hidden layers:
#### specifying the number of nodes, selecting an activation function, connecting the incoming layer
#### In hidden_layer1, the number of nodes is 128 and relu is the activation function. Notice the positioning of input_layer in brackets at the end. Bracket notations are used to specify where the input for each layer comes from.
#### Unlike Sequential API, where we simply stack layers together by calling one layer after the other, Functional API is more flexible, as we can create models with shared layers by making a call to the previous layer as a component of the current layer.

hidden_layer1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)

## Add Dropout layers:
#### Dropout layer is a regularization technique for neural networks that helps prevent overfitting.
####  The idea behind a dropout layer is to randomly drop units (either hidden or visible) from the network during training. A unit is a single neuron in a neural network. This has the effect of making the network more resistant to overfitting.

drop1 = tf.keras.layers.Dropout(rate=0.40)(hidden_layer1) # we add the first dropout layer, drop1, after the first hidden layer.
hidden_layer2 = tf.keras.layers.Dense(64, activation='relu')(drop1) # drop1 will become an input to second hidden layer, hidden_layer2

drop2 = tf.keras.layers.Dropout(rate=0.2)(hidden_layer2)
hidden_layer3 = tf.keras.layers.Dense(16, activation='relu')(drop2)

hidden_layer4 = tf.keras.layers.Dense(8, activation='relu')(hidden_layer3)

hidden_layer5 = tf.keras.layers.Dense(4, activation='relu')(hidden_layer4)

output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer5) # The last layer is a dense layer with a sigmoid activation function
####  Sigmoid is a mathematical function that can be used to map any real value onto a value between 0 and 1. This means that the output of the last layer is a probability distribution over the classes.

## Instansiate the model:
#### Instantiate a model with the specification of the input and output layers.

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.summary()

## Compiling the Model:
#### requires: optimizer, loss function, metrics

optimizer = tf.keras.optimizers.Adam(0.001) # using Adam optimizer with learning rate = 0.001 which controls how much the weights of the neural network model are updated during training.

loss = tf.keras.losses.BinaryCrossentropy() # Since we're solving a binary classification algorithm, we can use the binary cross entropy loss function, which is used to compute the loss for a binary classification problem.

metrics = ['accuracy'] # The accuracy evaluation metric is used to evaluate the performance of a binary classification model.

model.compile(loss=loss, metrics=metrics, optimizer=optimizer) # compiling the model

## Fitting/train the model:

model.fit(X_train, y_train, epochs= 150, verbose=0) # training the model on the training data 150 times

## Evaluate the Model:

print(model.evaluate(X_train, y_train)) # BinaryCrossentropy=0.33, accuracy=0.86
print(model.evaluate(X_test, y_test)) # BinaryCrossentropy=0.51, accuracy=0.81

#### We achieved an accuracy of 86% on the training set and 81% on the test data set. These values indicate that the performance of the model is good.

#### Also: remember that if the train accuracy is much higher than the test accuracy, it could mean that the model is overfitting on the training data. This means that the model has learned patterns in the training data that do not generalize well to new data. It's important to monitor both train and test accuracy during model development and tune parameters accordingly to avoid overfitting. If you're seeing a large discrepancy between train and test accuracy, it's worth investigating to try to understand why. If the model is overfitting, you may need to use regularization techniques to prevent this. In our case, we don't see a problem, as there is comparable accuracy for both training and test datasets.

