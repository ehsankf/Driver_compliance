# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.\

# https://realpython.com/python-histograms/
# https://github.com/glgh/w99-demo/tree/master/js
# traffic volumes and rates https://dot.ca.gov/programs/traffic-operations/census
import csv
import numpy as np
import pandas
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from scipy.integrate import quad
from scipy.stats import uniform, randint

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import sys
import os

mph_to_ftps = 5280. / 3600
np.random.seed(42)

# Load style file for plotting
custom_style_path = "science-ieee.mplstyle"
plt.style.use([custom_style_path])
DPI = 600

""""
 n, bins = np.histogram(distances)
 mids = (bins[1:] + bins[:-1]) / 2.
 mean = np.average(mids, weights=n)
 var = np.average((mids - mean) ** 2, weights=n)
 print(n, mids)
 print(len(bins))

 print(f'Hi, {name}') # Press Ctrl+F8 to toggle the breakpoint.
"""
# Set safety reduction factor (SRF)
SRF = 0.6
CMPR_SRF = 0.75
# Flodar name related to each SRF'
from main_figures import SRF_Dict, CMPR_SRF_Name, names, csv_names
"""
SRF_Name = {0.6: "AV-06", 0.75: "AV-075"}[SRF] # "06"
# Comparative SRF folder name
CMPR_SRF_Name = {0.6: "AV-06", 0.75: "AV-075"}[CMPR_SRF]
# Set csv file
names = {"06": 'SRF_06\\', "04": 'SRF_04\\', "AV-06": 'AV_SRF_06\\', "AV-075":"AV_SRF_075\\",
         "AV-FAR-06": "AV_FAR_SRF_06\\"}
csv_names = {"06": 'csv_data_12.csv', "04": 'csv_data_17.csv', "AV-06": 'csv_data_06.csv',
             "AV-075": 'csv_data_075.csv', "AV-FAR-06": 'csv_data_4.csv'}
"""
SRF_Name = SRF_Dict[SRF]
CMPR_SRF_Name = SRF_Dict[CMPR_SRF]
data_folder_name = os.path.join('Data', names[SRF_Name])
csv_file_name = os.path.join(data_folder_name, csv_names[SRF_Name])
# Set the folder to save the figures
folder = os.path.join("ML_Figures", names[SRF_Name])
if not os.path.exists(folder):
    os.makedirs(folder)

def input_data(csv_file_name=csv_file_name):
    NameMapDict = {'volume': 'Volume', 'VehsDC(1)': 'LMTs', 'MPR(0)': 'MPR', 'Car(1)': 'Truck-Ratio',
                   'VehsDelay(1)': 'Throughput', 'VehDelayDelay(1)': 'Delay', 'QueueDelayDC(1)': 'QueueDelay1',
                   'QueueDelayDC(2)': 'QueueDelay2', 'SpeedDC(1)': 'Speed1', 'SpeedDC(2)': 'Speed2',
                   'dist_distr': 'Distr.'}
    skip_rows = 0
    data = pd.read_csv(csv_file_name, skiprows=skip_rows)
    data.rename(columns=NameMapDict, inplace=True)
    data['Density'] = data['VehsDC(2)'] / data['Speed2']
    data['Distr.'] = data['Distr.'].apply(lambda x: 6 if x == 7 else (7 if x == 6 else x))
    indexer = data['MPR'] > 0
    data.loc[indexer, 'MPR'] = data[indexer]['MPR'] + 0.01
    return data



def sample_empirical_dist(distr, random_uniform_values):
    xs = [[50.0, 600.0, 600.0, 1200, 2500],
          [50.0, 600.0, 600.0, 1200.0, 2500],
          [50.0, 600.00, 600.0, 730.0, 880.0, 1020.0, 1080.0, 1120.0, 1200.0, 2500],
          [50.0, 600.0, 600.0, 1000, 1200.0, 2500],
          [50.0, 1200.0, 1200.0, 2500],
          [50.0, 1200.0, 1200.0, 2500.0],
          [50.0, 800.0, 1200.0, 2500.0],
          [50.0, 1200.0, 1200.0, 2300.0, 2500.0]
          ]
    ys = [[0.0, 0.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.2, 1.0, 1.0],
          [0.0, 0.0, 0.20, 0.22, 0.26, 0.39, 0.47, 0.60, 1.0, 1.0],
          [0.0, 0.0, 0.2, 0.2, 1.0, 1.0],
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.20, 1.0],
          [0.0, 0.0, 0.20, 1.0],
          [0.0, 0.0, 0.2, 0.2, 1.0]
          ]

    # Define a function that represents the interpolated curve
    def interpolated_function(x):
        return np.interp(x, xs[distr], ys[distr])

    # Compute the integral using the quad function
    integral_value, error = quad(interpolated_function, xs[distr][0], xs[distr][-1])

    # integral_value = 2500 - integral_value
    # Generate random uniform values for sampling
    cdf = ys[distr]
    # Random selection using np.random.choice
    # probs = np.diff(cdf, prepend=0)
    # sampled_values = np.random.choice(xs[distr], size=n_samples, p=probs)
    sampled_values = np.interp(random_uniform_values, cdf, xs[distr])
    sampled_values = np.sort(sampled_values)
    sampled_values = np.insert(sampled_values, 0, integral_value)
    return sampled_values


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def fit_regression(y_train, X_train):
    model = linear_model.LinearRegression()
    # Apply the lambda function to the array using numpy.vectorize
    y_train = np.vectorize(lambda x: x + 1 if x == 0 else x)(y_train)
    # Apply a logarithmic transformation to the response variable
    y_train_log = np.log(y_train)
    model.fit(X_train, y_train_log)
    return model


def fit_xgboost(y_train, X_train):
    # Create the XGBoost DMatrix for training and testing data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Set the hyperparameters for the XGBoost model
    params = {
        'objective': 'count:poisson',  # Use 'count:poisson' objective for Poisson regression
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'n_estimators': 117
    }
    xgb_model = xgb.XGBRegressor(objective='count:poisson')
    params = {
        "eta": uniform(0.05, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4)
    }

    params3 = {
        'eta': uniform(0.1, 0.2),
        'max_depth': randint(3, 6),
        'gamma': uniform(0, 0.02),
        'subsample': uniform(0.6, 0.8),
        'n_estimators': randint(100, 150)
    }

    # hyperparameter search
    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1,
                                n_jobs=1, return_train_score=True)

    search.fit(X_train, y_train)
    return search.best_estimator_

    report_best_scores(search.cv_results_, 1)

    # Train the XGBoost model
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    return model


def fit_glm_with_family(y_train, X_train, family_name):
    """
    Fit a Generalized Linear Model (GLM) using the specified family.

    Parameters:
        y : array-like
            The response variable.
        X : array-like
            The design matrix (feature matrix).
        family_name : str
            The name of the family representing the probability distribution for the response variable.
            Options: 'poisson', 'negativebinomial', 'gaussian', 'binomial', etc.

    Returns:
        results : statsmodels.genmod.generalized_linear_model.GLMResults
            The results of the GLM fitting.
    """
    # Convert the family_name to lowercase to handle case insensitivity
    family_name = family_name.lower()
    family_name = family_name.replace(" ", "")
    if family_name == "xgboost":
        return fit_xgboost(y_train, X_train)

    # Map family_name to the corresponding statsmodels family class
    family_map = {
        'poisson': sm.families.Poisson,
        'negativebinomial': sm.families.NegativeBinomial,
        'gaussian': sm.families.Gaussian,
        'binomial': sm.families.Binomial,
        # Add other family names and classes as needed
    }
    if family_name not in family_map:
        raise ValueError(f"Invalid family name '{family_name}'. Please choose from: {', '.join(family_map.keys())}")
    family_class = family_map[family_name]
    # Fit the GLM model using the specified family
    model = sm.GLM(y_train, X_train, family=family_class())
    if family_name == "poisson":
        model.family.variance = sm.families.varfuncs.Power(3.5)  # Best value: 3.5
    results = model.fit(maxiter=600)
    return results


def print_for_plots(X, specifier=".2f", V=5):
    sec_to_hr = 3600 / 300.
    for i in range(int(len(X) / V)):
        for j in range(V):
            print(f"{sec_to_hr * X[i * V + j]:{specifier}}", end=', ')
        print("\n")


def print_results(data, label, delays, qdelay, distributions, volumes):
    for i, dis in enumerate(distributions):
        for j, vol in enumerate(volumes):
            print(
                f"Distribution: {dis:d} Volume: {vol:d}  LMT: {label[i * len(volumes) + j]} Throughput: {data[i * len(volumes) + j][0]:.2f} Delay: {delays[i * len(volumes) + j]:.2f} \
            Q-delay (lane 1): {qdelay[i * len(volumes) + j]:.2f}")


def bar_plot(data, id_var, res_var, ylabel, name):
    from cycler import cycler
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Define a custom color cycle that includes orange
    custom_cycler = cycler(color=['orange', 'r', 'b', 'g'])

    current_cycler = plt.rcParams['axes.prop_cycle']
    updated_cycler = custom_cycler + cycler(linestyle=current_cycler.by_key()['linestyle'])

    # Set the updated cycler for the current axes
    plt.gca().set_prop_cycle(updated_cycler)

    # Apply the custom color cycle to the current axes
    plt.rcParams['axes.prop_cycle'] = custom_cycler
    g = sns.boxplot(x=id_var, y=res_var, data=data)

    plt.xlabel(id_var, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    fig.savefig(os.path.join(folder, f'{name}_{id_var}_{res_var}.png'))
    plt.close(fig)
    plt.rcParams['axes.prop_cycle'] = current_cycler
    return


def relative_error_histogram(data, id_var="methods", val_var=list() , name="relative_error"):
    # Plot the normalized cumulative distribution
    fig, ax = plt.subplots(1, 1)
    for index, row in data.iterrows():
        # Plot the cumulative distribution
        meth = row[id_var]
        error = row[val_var]
        error = np.sort(error)
        ax.plot(error, np.arange(1, len(error) + 1) / len(error), label=meth)

    # Add labels and title
    ax.set_xlabel('Relative Errors')
    ax.set_ylabel('Normalized Cumulative Distribution')
    ax.set_title('Normalized Cumulative Distribution of Relative Errors')
    ax.legend()

    # Show the plot
    ax.grid()
    plt.savefig(folder + name + ".png")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    volumes = [1800, 2000]  # [600, 800, 1500, 2500, 4000]
    Distributions = [1]  # [1, 2, 3, 4, 5, 6, 7, 8]
    df = input_data()
    # Filter over link behavior
    df['SRF'] = df['link_behav'].apply(lambda x: 10 if x == '7 : AV-WZ' else 0)
    # df = df[df['link_behav'].apply(lambda x: x != '7 : AV-WZ')]
    num_distance_distr = 8
    distance_distr = {}
    n_samples = 20
    random_uniform_values = np.random.rand(n_samples)
    for i in range(num_distance_distr):
        distance_distr[i] = sample_empirical_dist(i, random_uniform_values)
    # Create the dataset and the labels
    # TODO: That would be interesting to compare volumes and count of vehicles
    # data = df[['VehsDelay(1)', 'MPR(0)', 'Car(0)', 'SRF']].values
    # df = df[(df['Volume'] < 2000) & (df['MPR'] < 0.6)]
    df = df[(df['Volume'] < 2000)]
    data = df[['Volume', 'MPR', 'Car(0)', 'Truck-Ratio', 'SRF']].values
    dist_array = np.array(list(map(lambda key: distance_distr.get(key), df["Distr."] - 1)))
    data = np.concatenate([data, dist_array], axis=1)
    label = np.floor(df['LMTs'].values)

    # Create the train and test samples
    test_size = 0.2
    num_samples = len(label)
    # Shuffle the data and labels together
    indices = np.random.permutation(num_samples)
    data = np.array(data)[indices]
    label = np.array(label)[indices]
    split_index = int(num_samples * test_size)
    X_train, y_train, X_test, y_test = data[split_index:], label[split_index:], data[:split_index], label[:split_index]

    methods = ["Regression"] # ["Regression", "XG Boost", "Poisson", "Negative Binomial"]
    results = {
        "methods": list(),
    }
    df_res = pd.DataFrame(columns=results)
    train_col_header = [f"train {i}" for i in range(1, len(X_train)+1)]
    test_col_header = [f"test {i}" for i in range(1, len(X_test)+1)]
    df_res = pd.concat([df_res, pd.DataFrame(columns=train_col_header),
                        pd.DataFrame(columns=test_col_header)], axis=1)
    for meth in methods:
        trans = lambda y: (np.exp(y) if meth == "Regression" else y)
        if meth == "Regression":
            model = fit_regression(y_train, X_train)
        else:
            model = fit_glm_with_family(y_train, X_train, meth)
        index = len(df_res)
        df_res.loc[index, "methods"] = meth
        for X_samples, y_samples, header in zip([X_train, X_test], [y_train, y_test], [train_col_header, test_col_header]):
            y_pred = trans(model.predict(X_samples))
            mse_errors = mean_squared_error(y_samples, y_pred)
            rel_errors = np.abs(y_pred - y_samples) / y_samples
            df_res.loc[index, header] = rel_errors

    for name, header in zip(["train", "test"], [train_col_header, test_col_header]):
        df_res["mean_error_"+name] = df_res[header].mean(axis=1)
        df_res["std_error_"+name] = df_res[header].std(axis=1)
        merged_df = df_res[['methods', *header]].melt(id_vars=['methods'], value_vars=header,
                                                            value_name='error')
        bar_plot(data=merged_df, id_var="methods", res_var="error", ylabel="MSE", name=name+"_box_plot_error")
        relative_error_histogram(df_res, id_var="methods", val_var=header, name=name+"_relative_error")

    for index, row in df_res.iterrows():
        print(f'{df_res["methods"]}: Mean Err {df_res["mean_error_test"]}, Std Err {df_res["std_error_test"]}')

    pdb.set_trace()

    model = sm.GLM(label, data, family=sm.families.NegativeBinomial())  # Define the Poisson regression model
    # Fit the model to the data
    results = model.fit()

    pdb.set_trace()

    names = [['sample_data_1000/Lane_Change_Compliance_ablation_dist1_001.fzp',
              'sample_data_1000/Lane_Change_Compliance_ablation_dist1_002.fzp',
              'sample_data_1000/Lane_Change_Compliance_ablation_dist1_003.fzp',
              'sample_data_1000/Lane_Change_Compliance_ablation_dist1_004.fzp',
              'sample_data_1000/Lane_Change_Compliance_ablation_dist1_005.fzp'],
             ['sample_data_2000/Lane_Change_Compliance_ablation_dist1_001.fzp',
              'sample_data_2000/Lane_Change_Compliance_ablation_dist1_002.fzp',
              'sample_data_2000/Lane_Change_Compliance_ablation_dist1_003.fzp',
              'sample_data_2000/Lane_Change_Compliance_ablation_dist1_004.fzp',
              'sample_data_2000/Lane_Change_Compliance_ablation_dist1_005.fzp']
             ]

    pdb.set_trace()
    # Open a file for writing

    output_file = 'poisson_regression_summary.txt'
    with open(output_file, 'w') as f:
        # Redirect standard output to the file
        sys.stdout = f

        # Print the model summary
        print(results.summary())

    # Restore standard output
    sys.stdout = sys.__stdout__

    sys.exit()

    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    xs = [0.6, 0.65, 1.65, 2.65, 3.65, 4.65, 5.65, 6.65, 8.0]
    ys_cars = [0.0, 0.1, 0.43, 0.76, 0.89, 0.96, 1.0, 1.0, 1.0]
    ys_trucks = [0.0, 0.0, 0.2, 0.27, 0.57, 0.77, 0.9, 1.0, 1.0]
    ax1.plot(xs, ys_cars, linewidth=2.5, c='#539ecd', linestyle='-', marker='o', mfc='black')
    for i_x, i_y in zip(xs[:-1], ys_cars[:-1]):
        ax1.text(i_x + 0.02, i_y, f'({i_x}, {i_y})')

    ax1.set_title('Passenger Cars', fontsize=30, pad=5)
    ax2.set_title('Trucks', fontsize=30, pad=5)
    ax1.set_xlim([0.55, 8.0])
    ax1.fill_between(xs, ys_cars, color='#539ecd')
    ax1.set_xlabel(r"CC1", fontsize=30)
    ax1.set_ylabel(r"cumulative distribution", fontsize=30)

    ax2.plot(xs[1:], ys_trucks[1:], linewidth=2.5, c='#539ecd', linestyle='-', marker='o', mfc='black')
    for i_x, i_y in zip(xs[1:-1], ys_trucks[1:-1]):
        ax2.text(i_x + 0.02, i_y, f'({i_x}, {i_y})')

    ax2.set_xlim([0.55, 8.0])
    ax2.fill_between(xs, ys_trucks, color='#539ecd')
    ax2.set_xlabel(r"CC1", fontsize=30)
    ax2.set_xlabel(r"CC1", fontsize=30)
    ax2.set_ylabel(r"cumulative distribution", fontsize=30)

    fig.tight_layout()
    fig.savefig(folder + "Wiedemann_CC1!" + ".png")

    # number of conflicts versus distance
    # distances = [], conflicts = [], conflicts by obstacles = []

"""
sample sample from a distribution
import numpy as np

counts, bin_edges = np.histogram(bins[1], bins=20)
normalized_count = counts/sum(counts)

# Define the numbers and their corresponding probabilities
numbers = [1, 2, 3, 4, 5]
probabilities = [0.2, 0.3, 0.1, 0.15, 0.25]

# Generate random samples based on the probabilities
samples = np.random.choice(numbers, size=1000, p=probabilities)

# Print the samples
print(samples)

midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
"""
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# https://www.diva-portal.org/smash/get/diva2:1195623/FULLTEXT01.pdf

# Modeling and simulation of vehicle group lane-changingbehaviors in upstream segment of ramp areas under a connectedvehicle environment

# https://www.activetbooks.com/adjusting-driving-behavior
# https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html
