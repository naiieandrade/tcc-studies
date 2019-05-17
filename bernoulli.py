#%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from auxiliar_functions import calculate_median, calculate_variance


# with three variables is dirichlet distribution
def generate_dirichlet_distribution_with_three_variables(list_alpha, size):
    if len(list_alpha) != 3:
        print("This is not a three list alpha. There is ", len(list_alpha))
    else:
        dist = np.random.dirichlet(list_alpha, size)
        return dist


def plot_dirichlet_distribution_with_three_variables(distribution, size):
    dist_transpose = distribution.transpose()
    plt.barh(range(size), dist_transpose[0])
    plt.barh(range(size), dist_transpose[1], left=dist_transpose[0], color='g')
    plt.barh(range(size), dist_transpose[2], left=dist_transpose[0]+dist_transpose[1], color='r')
    plt.title("Dirichlet distribution")


def generate_dataframe_describe_with_median(distribution):
    df_transpose = pd.DataFrame(data=distribution.T)
    df = df_transpose.describe()
    df_median = pd.DataFrame(calculate_median(df)).T
    df_var = pd.DataFrame(calculate_variance(df)).T
    df_describe = df.append(df_median)
    df_describe.rename(index={0: 'median'}, inplace=True)
    df_describe = df_describe.append(df_var)
    df_describe.rename(index={0: 'variance'}, inplace=True)
    return df_describe


def plot_diference_std_median(df_describe):
    df_describe.loc[['std', 'median', 'variance']].plot.bar(rot=0)


def generate_dirichlet_distribution_with_categories(categories, comments, list_alpha):
    dist = np.random.dirichlet(list_alpha, (categories, comments))
    return dist


def plot_dirichlet_distribution_with_categories(distribution, size_sample):
    dist_transpose = distribution[0][:size_sample].T
    plt.barh(range(size_sample), dist_transpose[0])
    plt.barh(range(size_sample), dist_transpose[1], left=dist_transpose[0], color='g')
    plt.title("Dirichlet distribution")


def separate_agreed_votes_per_category(distribution, categories):
    if distribution.shape[0] == categories:
        group_categories = []
        for i in range(categories):
            string = 'c'
            string += str(i)
            string = distribution[:, :, 0][i]
            group_categories.append(string)
        return group_categories


def create_column_category(list_votes):
    list_df_categories = []
    for i in range(len(list_votes)):
        dataframe = "df"
        dataframe += str(i+1)
        dataframe = pd.DataFrame(list_votes[i])
        dataframe['category'] = (i+1)
        list_df_categories.append(dataframe)
    return list_df_categories
