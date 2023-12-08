import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

#Pre Processing 

#UCI has separate datasets for red & white
red_wine_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

wine_red = pd.read_csv(red_wine_url, sep=';')
wine_white = pd.read_csv(white_wine_url, sep=';')

wine_red["color_label"] = 0 # 0 denotes red wine
wine_white["color_label"] = 1 # 1 denotes white wine 

wine = pd.concat([wine_red,wine_white],ignore_index=True)

X = wine.drop("color_label", axis=1) 
y = wine["color_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)



# For each continuous features create a histogram to show their mean, median, mode, min and max

for feature in X.columns:
    plt.figure(figsize=(4, 4))
    plt.hist(X[feature], bins=20, edgecolor='black')
    plt.title(f'Histogram for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()


# For each feature, apply GMM to separate each feature in to two types of wine: red or white.

#GMM model and evaluation
n_components = 2
# means = 
# covariances =
# weights = 



# def gmm_predict:
    

# y_pred = gmm_predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')


# Draw histogram for each continuous feature, now you have to use two different colors for wine.
for feature in X_train.columns:
    plt.figure(figsize=(4, 4))
    
    plt.hist(X_train[feature][y_train == 0], bins=20, color='maroon',edgecolor='black')
    plt.hist(X_train[feature][y_train == 1], bins=20, color='lightgoldenrodyellow',
             edgecolor='black',alpha=0.5)
    
    plt.title(f'Histogram for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Use a standard Python GMM API call and compare the outcome with your results.

gmm_standard_api = GaussianMixture(n_components=2, random_state=42).fit(X_train)
gmm_labels_standard_api = gmm_standard_api.predict(X_train)

y_pred_standard = gmm.predict(X_test)

accuracy_standard_api = accuracy_score(y_test, y_pred_standard)
print(f'Accuracy of GMM Standard: {accuracy_standard_api}')
