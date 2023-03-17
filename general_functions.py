# Python file with all general functions for the ECG project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle
import json
from keras.models import model_from_json
import datetime

# Def to combine all CSV files in a directory into a single DataFrame
def combine_csv_files_to_df(dir_path):
    # Get all CSV files in the directory
    csv_files = [file for file in os.listdir(dir_path) if file.endswith('.csv')]

    df_list = [pd.read_csv(os.path.join(dir_path, file), header=None) for file in csv_files]
    combined_data = pd.concat(df_list, ignore_index=True,)

    # Display the combined DataFrame
    return(combined_data)

# Def to plot a 5 random instances of a given label
def plot_ecg_examples(df, label, n_examples=5):    
    filtered_df = df[df[187] == label]

    # choose 5 random instances from the filtered DataFrame
    random_instances = filtered_df.sample(n=n_examples, random_state=42)

    return random_instances

# Def to plot a heat map/histogram of a given label
def plot_hist(df, class_number,size,min_,bins):
    img=df.loc[df[187]==class_number].values
    img=img[:,min_:size]
    img_flatten=img.flatten()

    final1=np.arange(min_,size)
    for i in range (img.shape[0]-1):
        tempo1=np.arange(min_,size)
        final1=np.concatenate((final1, tempo1), axis=None)
    # print(len(final1))
    # print(len(img_flatten))
    plt.hist2d(final1,img_flatten, bins=(bins,bins),cmap=plt.cm.jet)
    plt.show()

def get_random_data_from_class(df, class_number, n_examples=5):
    filtered_df = df[df[187] == class_number]
    random_instances = filtered_df.sample(n=n_examples, random_state=42)
    return random_instances

def pairplot_pca(np_result_pca, input_data, title):
    # np_result_pca is the result of the PCA transformation
    # input_data is the original data
    df_pca = pd.DataFrame(np_result_pca)
    df_labels = pd.DataFrame(input_data[187])
    df_labels = df_labels.set_index(df_pca.index)
    # add the target variable to the dataframe

    result = pd.concat([df_pca, df_labels], axis=1)
    result.rename(columns={187: 'Class'}, inplace=True)


    # create a pairplot with the T-SNE components and the target variable
    g = sns.pairplot(data=result, hue=result.columns[-1], palette='Set1')
    g.fig.suptitle(title, y=1)

def plot_3D_pca(np_result_pca, input_data, n_classes, labels, title):
    # np_result_pca is the result of the PCA transformation
    # input_data is the original data
    # Create a 3D scatter plot of the first 3 principal components
    y = input_data[187]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'c', 'm']

    for i in range(n_classes):
        idx = np.where(y == i)
        #print(idx)
        ax.scatter(np_result_pca[idx,0], np_result_pca[idx,1], np_result_pca[idx,2], c=colors[i], label=labels[i])
    #x.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c='b', marker='o') #, marker='o'

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()
    plt.title(title)
    plt.show()

def mean_dur(df, class_name):
    
    duration = list()

    filtered_df = df[df[187]==class_name]
    filtered_df.reset_index(drop=True, inplace=True)

    for i in range(len(filtered_df)):
        # last col contains label, exclude element in col 187
        sample_no_label = filtered_df.loc[i][0:187]

        # count for empty cols from the back
        flag = True
        j = 0
        while flag:
            col_val = sample_no_label[186-j]
            # Stop on first non zero element
            if col_val != 0:
                flag = False
            else:
                j = j + 1
        
        dur = 187 - j
        duration.append(dur)
        
    mean_dur = round(sum(duration)/len(duration),2)
            
    return mean_dur

# Save the results of PCA to a csv file
def save_pca_results_to_csv(pca_model, pca_data, filename):
    # Transform the data
    pca_data_transformed = pca_model.transform(pca_data[pca_data.columns[:-1]])
    df_pca = pd.DataFrame(data = pca_data_transformed)
    df_labels = pd.DataFrame(pca_data.iloc[:,-1:])
    df_labels = df_labels.set_index(df_pca.index)
        #add the target variable to the dataframe

    df_pca = pd.concat([df_pca, df_labels], axis=1)

    # Rename every column to PC1, PC2, PC3, PC4, PC5 etc based on column index
    # Get number of columns
    n_cols = df_pca.shape[1] -1

    for i in range (n_cols):
        df_pca.rename(columns={i: 'PC' + str(i+1)}, inplace=True)

    # Rename the last column to Class
    # Get name last column
    last_col_name = df_pca.columns[-1]
    df_pca.rename(columns={last_col_name: 'Class'}, inplace=True)

    # # Save the dataframe to a csv file with the delimiter ';'
    df_pca.to_csv(filename, index=False, sep=";", decimal=",")

def save_model(model, model_name, model_type):
    """
    Saves a scikit-learn or Keras model along with its hyperparameters and weights.

    Parameters:
    model (object): A scikit-learn or Keras model object.
    model_name (str): The name to give to the saved model.
    model_type (str): The type of model being saved ('sklearn' or 'keras').

    Returns:
    None
    """
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    folder_name = model_name + '_' + timestamp
    # Create a directory to save the model files in
    if not os.path.exists(f'saved_models/{folder_name}'):
        os.makedirs(f'saved_models/{folder_name}')
    

    # Save the model weights
    if model_type == 'sklearn':
        pickle.dump(model, open(f'saved_models/{folder_name}/{model_name}.pkl', 'wb'))
    elif model_type == 'keras':
        # Save the model architecture as a JSON file
        model_json = model.to_json()
        with open(f'saved_models/{folder_name}/{model_name}.json', 'w') as json_file:
            json_file.write(model_json)

        # Save the model weights as an HDF5 file
        model.save_weights(f'saved_models/{folder_name}/{model_name}.h5')

    # Save the model hyperparameters as a JSON file
    hyperparameters = model.get_params()
    with open(f'saved_models/{folder_name}/{model_name}_hyperparameters.json', 'w') as json_file:
        json.dump(hyperparameters, json_file)
