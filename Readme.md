# Classification Algorithms on the Make Moons Dataset

A hands-on machine learning project comparing multiple classification algorithms on the **`make_moons` dataset** from Scikit-learn.

The main goal of this project is to understand how different classification algorithms handle a **non-linearly separable dataset**, and to visualize their predictions and misclassified samples.

##  Project Overview

The `make_moons` dataset consists of two interleaving half-circle classes, making it a useful dataset for demonstrating the limitations and strengths of different classification techniques.

In this project, the following classifiers are implemented and compared:

* Support Vector Machine (SVM) with an RBF kernel
* K-Nearest Neighbours (K-NN)
* Decision Tree
* Random Forest
* Gaussian Naive Bayes

For each classifier, predictions are evaluated against the actual test labels, and misclassified points are visualized where applicable.

## Dataset

The dataset is generated using Scikit-learn's `make_moons()` function:

```python
X, y = make_moons(
    n_samples=100,
    noise=0.1,
    random_state=42
)
```

### Dataset characteristics

* **Samples:** 100
* **Features:** 2
* **Classes:** 2
* **Noise:** 0.1
* **Random state:** 42

The two features can be visualized as a 2D scatter plot, making this dataset particularly useful for understanding classification boundaries.

##  Algorithms Used

### 1. Support Vector Machine

An SVM classifier with an **RBF (Radial Basis Function) kernel** is used:

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma=0.7
)
```

The RBF kernel allows the SVM to create a **non-linear decision boundary**, making it well suited to the curved structure of the `make_moons` dataset.

---

### 2. K-Nearest Neighbours

K-NN is implemented with:

```python
KNeighborsClassifier(n_neighbors=3)
```

Instead of explicitly learning a mathematical decision boundary, K-NN classifies a new point based on the labels of its nearest neighbours.

This makes it naturally capable of handling non-linear class boundaries.

---

### 3. Decision Tree

A Decision Tree classifier is used with entropy as the splitting criterion:

```python
DecisionTreeClassifier(
    criterion='entropy',
    random_state=100,
    max_depth=3,
    min_samples_leaf=5
)
```

The tree creates a series of feature-based splits to separate the classes.

---

### 4. Random Forest

A Random Forest classifier is also tested:

```python
RandomForestClassifier()
```

Random Forest combines multiple decision trees and aggregates their predictions.

This generally produces a more robust classifier than relying on a single decision tree.

A **confusion matrix** is also generated to examine the classification results.

---

### 5. Gaussian Naive Bayes

Finally, Gaussian Naive Bayes is applied:

```python
GaussianNB()
```

Gaussian Naive Bayes assumes that the features follow Gaussian distributions within each class and that the features are conditionally independent given the class.

A 5-fold cross-validation score is also calculated:

```python
scores = cross_val_score(model, X, y, cv=5)
```

##  Misclassification Analysis

For several classifiers, the project identifies incorrectly classified test samples using:

```python
misclassified_mask = (y_test != y_predicted)
```

The coordinates of these points are then extracted and highlighted on the plots.

The visualizations compare:

* **Predicted Labels**
* **Actual Labels**
* **Misclassified Points**

This makes it easier to understand not only the accuracy of each model, but also **where and how the models make mistakes**.

##  Visualizations

The project generates visualizations for:

1. The original `make_moons` dataset
2. SVM predictions
3. K-NN predictions
4. Decision Tree predictions
5. Random Forest predictions
6. Naive Bayes predictions
7. Misclassified samples
8. Random Forest confusion matrix

The predicted and actual classifications are displayed side-by-side for easier comparison.

## Project Structure

```text
.
├── README.md
└── classification.ipynb
```

> Rename the notebook in the structure above if your actual notebook has a different filename.

##  Requirements

The project uses Python and the following libraries:

```text
numpy
scikit-learn
matplotlib
seaborn
```

Install the required packages using:

```bash
pip install numpy scikit-learn matplotlib seaborn
```

##  How to Run

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Navigate into the repository:

```bash
cd <repository-name>
```

3. Install the dependencies:

```bash
pip install numpy scikit-learn matplotlib seaborn
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook
```

5. Run the cells sequentially to generate the dataset, train the classifiers, evaluate their performance, and visualize their predictions.

##  Learning Objectives

This project was created to explore:

* Classification on non-linear datasets
* Train/test splitting
* Model fitting and prediction
* Accuracy evaluation
* Cross-validation
* Confusion matrices
* Misclassification analysis
* Visualization of classification results
* Differences between distance-based, tree-based, probabilistic, and kernel-based classifiers

##  Key Takeaway

The `make_moons` dataset is a simple but useful demonstration of why **the choice of model matters**.

A dataset may not be linearly separable, but that does not mean it cannot be classified effectively. Different algorithms approach the problem in different ways:

| Algorithm            | Main Idea                      | Handles Non-Linearity           |
| -------------------- | ------------------------------ | ------------------------------- |
| SVM + RBF            | Kernel-based decision boundary | ✅                               |
| K-NN                 | Neighbour-based classification | ✅                               |
| Decision Tree        | Recursive feature splits       | ✅                               |
| Random Forest        | Ensemble of decision trees     | ✅                               |
| Gaussian Naive Bayes | Probabilistic Gaussian model   | ⚠️ Depends on data distribution |

The project therefore focuses not only on **which model performs better**, but also on understanding **why their predictions differ**.

##  Possible Future Improvements

Some possible extensions to this project are:

* Compare training and testing accuracy for every model
* Plot decision boundaries for each classifier
* Tune hyperparameters using `GridSearchCV`
* Add Logistic Regression as a linear baseline
* Add a linear SVM to demonstrate the limitation of linear decision boundaries
* Compare precision, recall, and F1-score
* Experiment with different levels of noise
* Increase the number of samples
* Compare model performance using multiple random train/test splits
* Add XGBoost or other ensemble methods

##  Technologies

* **Python**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

---

### Author

**Manank Agarwal**

This project is part of my hands-on exploration of machine learning classification algorithms and their behaviour on different types of datasets.
