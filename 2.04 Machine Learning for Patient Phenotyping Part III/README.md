# Machine learning for patient stratification and classification

This workshop was developed for the UP-MIT Critical Data-Stanford Big Data for Health Workshops and Conference.

The workshop will introduce the audience to machine learning techniques. Participants will explore the differences between supervised and unsupervised learning and practice implementing them using Python and the SciPy ecosystem. It will focus on patient stratification and classification, for decision support in the intensive care unit.

# Getting Started

## Prerequisites
* Anaconda
* Knowledge of basic python syntax

## Installing
This project requires Python and the following Python libraries installed:

* **NumPy**: fundamental package for scientific computing with Python. Provides a fast numerical array structure.
* **pandas**: provides high-performance, easy-to-use data structures and data analysis tools.
* **scikit-learn**: essential machine learning package in Python. Provides simple and efficient tools for data mining and data analysis.
* **matplotlib**: basic plotting library.
* **seaborn**: visualization library based on matplotlib. It provides a high-level interface for drawing statistical graphics.
* **IPython**: for interactive data visualization.
* **Jupyter**: for interactive computing.

It is highly recommended that you install [Anaconda](https://www.anaconda.com/download/#macos), which already has the above packages and more included. Additionally, in order to be able to visualize decision trees, you should install [pydot](https://anaconda.org/anaconda/pydot) and [graphviz](https://graphviz.gitlab.io/download/).

## Running

The majority of this workshop will be taught through the use of a Jupyter notebook, which you can download from [here](https://github.com/cmsalgado/workshop-bd4h/blob/master/ML_manilla.ipynb). Alternatively, you can clone the repository using git:

> git clone https://github.com/cmsalgado/workshop-bd4h/blob/master/ML_manilla.ipynb

A csv file containing the data used for this workshop can be found [here](https://www.dropbox.com/s/3l20njo1sb5ui4g/mimic3_mortality.csv?dl=0).

# Data

The data were extracted from the MIMIC-III Clinical Database, which is a large, publicly-available database comprising de-identified electronic health records of approximately 60,000 ICU admissions. Patients stayed in the intensive care unit (ICU) of the Beth Israel Deaconess Medical Center between 2001 and 2012. MIMIC-III database is described in:

> Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, Mark RG. MIMIC-III, a freely accessible critical care database. Scientific Data (2016).

The extracted data is stored in a csv file containing 1,730,668 rows and 17 columns. Each row corresponds to a set of observations obtained at a specific time in the ICU, and each row is associated with a ICU stay ID, which is unique for every patient. There are 21,140 unique ICU stays. The dataset was constructed based on the code provided [here](https://github.com/YerevaNN/mimic3-benchmarks) for the prediction of hospital mortality using early ICU admission data (first 48 hours).

**Variables:**

* Age
*	Gender
*	Weight
*	Height
*	Diastolic blood pressure
*	Systolic blood pressure
*	Mean blood pressure
*	Temperature
*	Respiratory rate
*	Glucose
*	pH
*	Glasgow coma scale

**Target variable:**

* In-hospital mortality

# Topics covered

Machine learning is divided in two main parts:
* Patient stratification through unsupervised learning
  * Visualization of time series trends partitioned by outcome
  * k-means clustering
* Patient classification through supervised learning
  * k-nearest neighbors
  * Logistic regression
  * Decision trees
  * Random forest

Other topics include:

* Data preprocessing
  * Outliers
  * Missing data

* Feature engineering
  * Feature construction
  * Feature selection
  * Normalization

* Model evaluation
  * ROC, AUC, sensitivity, specificity
  * Threshold optimization 

* Handling overfitting
  * Motivation to bias-variance
  * Training and testing

# Authors
Cátia Salgado

# Acknowledgments
Nikhil Shankar for providing the clinical input
