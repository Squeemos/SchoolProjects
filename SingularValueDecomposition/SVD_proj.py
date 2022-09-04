import streamlit as st
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

import tensorflow as tf

import pandas as pd
import cv2
import numpy as np
import os

# Constants
data_path = "./leedsbutterfly_dataset_v1.0/leedsbutterfly/images/"
target_names = ["Danaus plexippus", "Heliconius charitonius", "Heliconius erato", "Junonia coenia", "Lycaena phlaeas", "Nymphalis antiopa", "Papilio cresphontes", "Pieris rapae", "Vanessa atalanta", "Vanessa cardui"]
n_classes = 10


@st.cache(suppress_st_warning=True)
def load_images(path,channel,image_size):
    targets,imgs = [],[]
    for image in os.listdir(path):
        targets.append(int(image[:3]) - 1)
        img = plt.imread(data_path + image)
        img = cv2.resize(img,image_size)
        imgs.append(img[:,:,channel].flatten())

    return np.array(targets), np.array(imgs)

@st.cache(suppress_st_warning=True)
def run_pca(n,x_train,x_test):
    pca = PCA(n,svd_solver='full')
    pca.fit(x_train)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    return pca,x_train_pca,x_test_pca

@st.cache(suppress_st_warning=True)
def run_svm(x_train_pca, x_test_pca, y_train, y_test):
    param_grid = {"C": [1e3, 5e3, 1e4, 5e4, 1e5],"gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
    clf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid)
    clf = clf.fit(x_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    report = classification_report(y_test, y_pred, target_names=target_names,output_dict=True)

    return clf,y_pred,report

@st.cache(suppress_st_warning=True)
def run_decision_tree(x_train_pca,x_test_pca,y_train,y_test):
    DT = DecisionTreeClassifier()
    DT.fit(x_train_pca,y_train)
    y_pred = DT.predict(x_test_pca)

    report = classification_report(y_test,y_pred,target_names=target_names,output_dict=True)

    return DT,y_pred,report

@st.cache(suppress_st_warning=True)
def run_knn(x_train_pca,x_test_pca,y_train,y_test,neighbors):
    kn = KNeighborsClassifier(neighbors)
    kn.fit(x_train_pca,y_train)
    y_pred = kn.predict(x_test_pca)

    report = classification_report(y_test,y_pred,target_names=target_names,output_dict=True)

    return kn,y_pred,report

@st.cache(suppress_st_warning=True)
def run_random_forest(x_train_pca,x_test_pca,y_train,y_test):
    forest = RandomForestClassifier()
    forest.fit(x_train_pca,y_train)
    y_pred = forest.predict(x_test_pca)

    report = classification_report(y_test,y_pred,target_names=target_names,output_dict=True)
    return forest,y_pred,report

@st.cache(suppress_st_warning=True)
def run_gbc(x_train_pca,x_test_pca,y_train,y_test):
    gbc = GradientBoostingClassifier()
    gbc.fit(x_train_pca,y_train)
    y_pred = gbc.predict(x_test_pca)

    report = classification_report(y_test,y_pred,target_names=target_names,output_dict=True)
    return gbc,y_pred,report

def run_neural_net(x_train_pca,x_test_pca,y_train,y_test,img_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=img_size),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(x_train_pca, y_train, epochs=20)
    test_loss, test_acc = model.evaluate(x_train_pca, y_train, verbose=2)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(x_test_pca)
    y_pred = np.array([np.argmax(predict) for predict in predictions])
    report = classification_report(y_test,y_pred,target_names=target_names,output_dict=True)

    return model,y_pred,report


page = st.sidebar.selectbox("What project would you like to see", ("PCA with Butterflies",))

if page == "PCA with Butterflies":
    i_s = st.sidebar.text_input("Enter the size of the image",value="320,320")
    image_size = tuple(map(int, i_s.split(',')))

    c = st.sidebar.selectbox("Which color channel would you like this to run on?",("Red","Green","Blue"))
    n_components = st.sidebar.number_input("Enter the number of components to decompose into", value=100)
    test_size = float(st.sidebar.text_input("Enter the size of the testing data (as a decimal)",value=".25"))
    random_state = st.sidebar.number_input("Enter the random state to split the data by", value=7)
    classifier = st.sidebar.selectbox("What classifier would you like to use", ("SVM", "Decision Tree", "KNN", "Random Forest", "Gradient Tree Boost", "Neural Network"))

    channel = 0 if c == "Red" else (1 if c == "Green" else (2 if c == "Blue" else -1))
    targets,images = load_images(data_path,channel,image_size)

    X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=test_size, random_state=random_state)

    pca,X_train_pca,X_test_pca = run_pca(n_components,X_train,X_test)

    eigen_butterflies = pca.components_.reshape((n_components,*image_size))

    if classifier == "SVM":
        clf,y_pred,report = run_svm(X_train_pca,X_test_pca,y_train,y_test)
    elif classifier == "Decision Tree":
        DT, y_pred, report = run_decision_tree(X_train_pca,X_test_pca,y_train,y_test)
    elif classifier == "KNN":
        n = st.sidebar.number_input("Enter the number of neighbors for KNN",value=5,min_value=1)
        kn,y_pred,report = run_knn(X_train_pca,X_test_pca,y_train,y_test,n)
    elif classifier == "Random Forest":
        forest,y_pred,report = run_random_forest(X_train_pca,X_test_pca,y_train,y_test)
    elif classifier == "Gradient Tree Boost":
        gbc,y_pred,report = run_gbc(X_train_pca,X_test_pca,y_train,y_test)
    elif classifier == "Neural Network":
        x_train_reshape = X_train.reshape(624,*image_size)
        x_test_reshape = X_test.reshape(208,*image_size)
        model,y_pred,report = run_neural_net(x_train_reshape,x_test_reshape,y_train,y_test,image_size)


    df = pd.DataFrame(report).transpose()
    st.dataframe(df,800,800)

    img_view = st.sidebar.slider("Select which eigen butterfly you would like to view", value=0,min_value=0,max_value=n_components-1)
    scaler = MinMaxScaler()
    flat = eigen_butterflies[img_view].reshape(-1,1)
    scaler.fit(flat)
    flat = scaler.transform(flat)
    flat = flat.reshape(image_size)
    st.image(flat,width=640)
