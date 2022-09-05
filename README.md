# This repository is a collection of several school projects
**Please do not copy or redistribute code from this repository.**

### [Expectation Maximization](https://github.com/Squeemos/SchoolProjects/tree/main/ExpectationMaximization)
This project was to implement and work with with Gaussian Mixture Models and Expectation Maximization. Additional parts of this project were to write our own implementation of KMeans++

### [Fuzzy Sets](https://github.com/Squeemos/SchoolProjects/tree/main/FuzzySets)
#### Contents
- A "library" for:
    - Creating fuzzy sets of different types
        - Empty
        - Singleton
        - Interval
        - Triangular
        - Trapozoidal
        - Custom function
    - Implication of two fuzzy sets with a function
    - Creation of a Takagi-Sugeno Fuzzy Inference System (FIS) with custom output rules
    - Ability to approximate a function using a Takagi-Sugeno FIS
- Some examples of the "library" including:
    - Using Lukasiewicz, Larson, and Gorguen implication
    - Creating a Takagi-Sugeno FIS with sample output rules
    - Approximation of several example functions with a Takagi-Sugeno FIS and plotting of it

### [Gaussian Naive Bayes](https://github.com/Squeemos/SchoolProjects/tree/main/GaussianNaiveBayes)
This project was implementation Gaussian Naive Bayes (GNB) on the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)

#### Contents
- A helper function for computing Naive Bayes
- Function that predicts the label based on GNB for a testing dataset based on some targets from a training dataset (ground truth)
- Comparison between my implementation and the implementation from SciKitLearn
- Same comparison, but after some data preprocessing:
    - Removal of outliers with a z-score over 3
    - Removal of duplicates
    - Testing with different training/testing sizes

### [ID3 Decision Tree](https://github.com/Squeemos/SchoolProjects/tree/main/ID3DecisionTree)
This project was to implement an ID3 Decision Tree on a Tennis dataset and a Personal Loan dataset

#### Contents
- Helper functions for:
    - Calculating entropy from a list
    - Calculating the total entropy of a dataset
    - Calculating the entropy of a specific attribute
- Creating the ID3 tree
- Classifying a pandas Series from an ID3 tree
- Computing the accuracy of a tree with sample dataframe
- Visualizing the ID3 tree [from this stack overflow implementation](https://stackoverflow.com/questions/13688410/dictionary-object-to-decision-tree-in-pydot)
- Computation of accuracy on Tennis dataset with sample points
- Computation of accuracy on Personal Loan dataset after some preprocessing
    - Removal of duplicates
    - Removal of statistical outliers
- Computation of accuracy on Personal Loan dataset with different training sizes
- Computation of accuracy on Personal Loan dataset with different maximum tree depth

### [LSGAN](https://github.com/Squeemos/SchoolProjects/tree/main/LSGAN)
This project was an implementation of a Least-Squares Generative Adversarial Network (LSGAN) to generate cat faces from [this cat face dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models)

### [Pokemon Dataset Analysis](https://github.com/Squeemos/SchoolProjects/tree/main/PokemonDataAnalysis)

### [SVM and Agglomerative Clustering](https://github.com/Squeemos/SchoolProjects/tree/main/SVMandAgglomerativeClustering)

### [SVD with images](https://github.com/Squeemos/SchoolProjects/tree/main/SingularValueDecomposition)

### [Spam Filter with Gaussian Naive Bayes](https://github.com/Squeemos/SchoolProjects/tree/main/SpamFilter)

### [Stock Price Analysis](https://github.com/Squeemos/SchoolProjects/tree/main/StockPriceAnalysis)

### [Time Series Analysis in R](https://github.com/Squeemos/SchoolProjects/tree/main/TimeSeriesAnalysis)
