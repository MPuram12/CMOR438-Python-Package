# CMOR438-Python-Package

Authors: Mani Puram (mp108), Jonathan Cheng (jc212), Kenneth Soh (ks185)

# My Machine Learning Algorithms Package (e.g., ml_algorithms)

This Python package implements various machine learning algorithms, organized into supervised and unsupervised learning categories.

## Package Structure

The package is divided into 2 subfolders: ```supervised``` and ```unsupervised```. 

The ```supervised``` folder has 8 machine learning algorithms:
1. Perceptron
2. Neural Networks
3. Linear Regression
4. Logistic Regression
5. K-Nearest Neighbors
6. Decision Trees
7. Random Forests
8. Boosting

The ```unsupervised``` folder has 5 machine learning algorithms:
1. k-means Clustering
2. DBSCAN (Density Based Clustering)
3. Principal Components Analysis
4. SVD Decomposition for Image Compression
5. Label Propagation for Graphs
6. Maximum Clique for Graphs


## Instructions for Use


Within a Jupyter Notebook, import in the specific Machine Learning Model at the top. For example, if you wanted to use Decision Trees, which is in the ```supervised``` folder, you could include the line:

```from my_ml_package.supervised.decision_trees import DecisionTreeRegressor```

If you want to see each model in action applied to a very simple example, you can exeucte each ```.py``` file directly. 




