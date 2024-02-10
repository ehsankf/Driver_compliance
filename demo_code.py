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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.inspection import permutation_importance

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
from main_figures import SRF_Dict, names, csv_names, unit_conversion
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

def reshap(arr: np.array) -> np.array:
    if len(arr.shape) == 1:
        return arr.reshape(len(arr), 1)
    return arr
def input_data(file_name=csv_file_name):
    NameMapDict = {'volume': 'Volume', 'VehsDC(1)': 'LMTs', 'MPR(0)': 'MPR', 'Car(1)': 'Truck-Ratio',
                   'VehsDelay(1)': 'Throughput', 'VehDelayDelay(1)': 'Delay', 'QueueDelayDC(1)': 'QueueDelay1',
                   'QueueDelayDC(2)': 'QueueDelay2', 'SpeedDC(1)': 'Speed1', 'SpeedDC(2)': 'Speed2',
                   'dist_distr': 'Distr.'}
    skip_rows = 0
    data = pd.read_csv(file_name, skiprows=skip_rows)
    data.rename(columns=NameMapDict, inplace=True)
    data['Density'] = data['VehsDC(2)'] / data['Speed2']
    data['Distr.'] = data['Distr.'].apply(lambda x: 6 if x == 7 else (7 if x == 6 else x))
    indexer = data['MPR'] > 0
    data.loc[indexer, 'MPR'] = data[indexer]['MPR'] + 0.01
    data['MPR'] = data['MPR'].apply(lambda x: int(x * 100))
    # Filter over link behavior
    data['SRF'] = [SRF if file_name == csv_file_name else CMPR_SRF] * len(data)
    data['Intercept'] = [1] * len(data)
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
    # sampled_values = np.insert(sampled_values, 0, integral_value)
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
    model.fit_intercept = False
    # Apply the lambda function to the array using numpy.vectorize
    y_train = np.vectorize(lambda x: x + 1 if x == 0 else x)(y_train)
    # Apply a logarithmic transformation to the response variable
    y_train_log = np.log(y_train)
    model.fit(X_train, y_train_log)
    return model



def fit_random_forest(y_train, X_train, params="fixed"):

    if params is None:
        rf_regressor = RandomForestRegressor()
        params = {
            'n_estimators': [50, 100, 200],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees (None means unlimited)
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'max_features': ['auto', 'sqrt'],  # Number of features to consider for the best split
            'bootstrap': [True, False],  # Whether to use bootstrapped samples when building trees
            'criterion': ['mse', 'mae']  # Splitting criterion ('mse' for mean squared error, 'mae' for mean absolute error)
        }
        # hyperparameter search
        search = RandomizedSearchCV(rf_regressor, param_distributions=params, random_state=42, n_iter=200, cv=3,
                                    verbose=1,
                                    n_jobs=1, return_train_score=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        params = {'n_estimators': 200, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1,
                  "criterion": "mse", "max_features": "auto"}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
    return model

def fit_xgboost(y_train, X_train, params="fixed"):

    if params is None:
        xgb_model = xgb.XGBRegressor()
        params = {
            "eta": uniform(0.05, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 0.3),  # default 0.1
            "max_depth": randint(2, 6),  # default 3
            "n_estimators": randint(100, 150),  # default 100
            "subsample": uniform(0.6, 0.4),
            "objective": ["reg:squarederror", "reg:squaredlogerror", "count:poisson", "reg:gamma"]
        }
        # hyperparameter search
        search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1,
                                n_jobs=1, return_train_score=True)

        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        params = {'objective': 'count:poisson', 'base_score': 0.5, 'gamma': 0.042258503840340966,
         'learning_rate': 0.20918342467662915, 'max_delta_step': 0.699999988,
         'max_depth': 5, 'n_estimators': 114, 'predictor': 'auto', 'eta': 0.310490332110676}
        # Set the hyperparameters for the XGBoost model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
    return model

def fit_glm_with_family(y_train, X_train, family_name,
                        data=None, group_var='X5', params="fixed"):
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
    if family_name == "rfregressor":
        return fit_random_forest(y_train, X_train, params)
    if family_name == "xgboost":
        return fit_xgboost(y_train, X_train, params)

    # Map family_name to the corresponding statsmodels family class
    family_map = {
        'poisson': sm.families.Poisson,
        'negativebinomial': sm.families.NegativeBinomial,
        'gaussian': sm.families.Gaussian,
        'binomial': sm.families.Binomial,
        'mix': sm.MixedLM.from_formula
        # Add other family names and classes as needed
    }
    if family_name not in family_map:
        raise ValueError(f"Invalid family name '{family_name}'. Please choose from: {', '.join(family_map.keys())}")
    family_class = family_map[family_name]
    if family_name == "mix":
        formula = 'X0 ~ ' + ' + '.join(X_train.columns)
        model = family_class(formula, data=data, groups=group_var)
        results= model.fit()
        return results

    # Fit the GLM model using the specified family
    model = sm.GLM(y_train, X_train, family=family_class())
    # if family_name == "poisson":
    #    model.family.variance = sm.families.varfuncs.Power(3.5)  # Best value: 3.5
    results = model.fit(maxiter=600)
    return results


def data_split(df, featVec, res_var, test_size):
    data = df[featVec].values
    dist_array = np.array(list(map(lambda key: distance_distr.get(key), df["Distr."] - 1)))
    data = np.concatenate([data, dist_array], axis=1)
    label = np.floor(df[res_var].values)
    label += 1
    # Create the train and test samples
    test_size = 0.2
    num_samples = len(label)
    # Shuffle the data and labels together
    indices = np.random.permutation(num_samples)
    data = np.array(data)[indices]
    label = np.array(label)[indices]
    split_index = int(num_samples * test_size)
    data_col_header = [f"X{i}" for i in range(1, len(data[0])+1)]
    label_col_header = [f"X{0}"]
    X_train, y_train = pd.DataFrame(data[split_index:], columns=data_col_header), \
        pd.DataFrame(label[split_index:], columns=label_col_header)
    X_test, y_test = pd.DataFrame(data[:split_index], columns=data_col_header), \
        pd.DataFrame(label[:split_index], columns=label_col_header)
    return X_train, y_train, X_test, y_test

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

def heat_map_plot(data, id_var1, id_var2, res_var,
                  id_var3, spec_val3, id_var4, spec_val4, name, return_data=False):
    data = data[(data[id_var3]==spec_val3) & (data[id_var4]==spec_val4)]
    data[res_var] = data[res_var].apply(lambda x: int(x))
    heatmap_data = pd.pivot_table(data, values=res_var, index=[id_var2], columns=id_var1)
    heatmap_data = heatmap_data.sort_values(by=id_var2, ascending=False)
    g = sns.heatmap(heatmap_data, linewidths=0.25, cmap="coolwarm")
    g.tick_params(axis='both', which='both', top=False, right=False)
    # Remove minor tick labels
    g.set_xticks([], minor=True)
    g.set_yticks([], minor=True)
    # Remove the minor ticks from the colorbar
    cbar = g.collections[0].colorbar
    cbar.minorticks_off()
    # Set the labels for x-axis, y-axis
    plt.xlabel(unit_conversion[id_var1])
    units = {"MPR": "MPR (%)", "Volume": "Traffic Demand (vph/2 lanes)"}
    plt.ylabel(units[id_var2])
    plt.title(unit_conversion[res_var])
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'{name}_{id_var1}_{id_var2}.png'))
    plt.close()
    return heatmap_data

def heat_map_two_subplot(data1, data2, xlabels, ylabels, tlabels, name):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        units = {"MPR": "MPR (%)", "Volume": "Traffic Demand (vph/2 lanes)"}
        for ax, data, id_var1, id_var2, res_var in zip(axs, [data1, data2], xlabels, ylabels, tlabels):
            g = sns.heatmap(data, linewidths=0.25, cmap="coolwarm", ax=ax, cbar=False)
            g.tick_params(axis='both', which='both', top=False, right=False, labelsize=14)
            # Remove minor tick labels
            g.set_xticks([], minor=True)
            g.set_yticks([], minor=True)
            ax.set_xlabel(unit_conversion[id_var1], fontsize=18)
            ax.set_ylabel(units[id_var2], fontsize=18)
            ax.set_title(unit_conversion[res_var], fontsize=18)

        """
        # Remove the minor ticks from the colorbar
        cbar = g.collections[0].colorbar
        cbar.minorticks_off()
        # Set the labels for x-axis, y-axis
        plt.xlabel(unit_conversion[id_var1])
        units = {"MPR": "MPR (%)", "Volume": "Traffic Demand (vph/2 lanes)"}
        plt.ylabel(units[id_var2])
        plt.title(unit_conversion[res_var])
        """
        # Add a colorbar for the heatmaps
        cbar_ax = fig.add_axes([1, 0.125, 0.03, 0.79])  # Adjust the position and size of the colorbar
        cbar = fig.colorbar(ax.collections[0], cax=cbar_ax)
        # Set the size of tick labels on the colorbar
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f'{name}.png'))
        plt.close()


def bar_plot(data, id_var, res_var, ylabel, name, hu_var=None, type="box"):
    from cycler import cycler
    # Define a custom color cycle that includes orange
    custom_cycler = cycler(color=['orange', 'r', 'b', 'g'])

    current_cycler = plt.rcParams['axes.prop_cycle']
    updated_cycler = custom_cycler + cycler(linestyle=current_cycler.by_key()['linestyle'])

    # Set the updated cycler for the current axes
    plt.rcParams['axes.prop_cycle'] = updated_cycler

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6.3, 5.5))
    # sns plot styles: https://seaborn.pydata.org/tutorial/categorical.html
    func = sns.boxplot if type == "box" else sns.barplot
    g = func(x=id_var, y=res_var, hue=hu_var, data=data, color='b')
    methods = ["Log Linear\nRegression", "Negative\nBinomial", "RF Regressor", "XG Boost"]
    plt.xticks([0, 1, 2, 3], labels=methods, fontsize=15)
    plt.xlabel("Method")
    plt.ylabel(ylabel)
    ax.minorticks_off()
    ax.grid()
    plt.tight_layout()
    fig.savefig(os.path.join(folder, f'{name}_{id_var}_{res_var}.png'))
    plt.close(fig)
    return


def relative_error_histogram(data, id_var="Methods", val_var=list() , name="relative_error"):
    # Plot the normalized cumulative distribution
    fig, ax = plt.subplots(1, 1, figsize=(6.3, 5.5))
    for index, row in data.iterrows():
        # Plot the cumulative distribution
        meth = row[id_var]
        error = row[val_var]
        error = np.sort(error)
        ax.plot(error, np.arange(1, len(error) + 1) / len(error), label=meth)

    # Add labels and title
    ax.set_xlabel(r'$(\hat{y}_i - y_i)\,/\,y_i$')
    ax.set_ylabel('Normalized Cumulative Distribution')
    ax.legend()

    # Show the plot
    ax.grid()
    plt.savefig(folder + name + ".png")

def feature_importance(methods, feature_names=None, name="FeatureImportance"):
    feature_scores = 0
    if isinstance(feature_names, list):
        feature_names = np.array(feature_names)
    for (meth, model) in methods:
        for attr in ['feature_importances_', 'params', 'coef_']:
            if hasattr(model, attr):
                feature_scores = getattr(model, attr)
                if attr == "coef_":
                    feature_scores = feature_scores[0]
                sorted_idx = feature_scores.argsort()
                fig, ax = plt.subplots(1, 1)
                if attr == 'feature_importances_':
                    inter_ind = np.where(feature_names == "Intercept")
                    ax.barh(np.delete(feature_names[sorted_idx], inter_ind),
                            np.delete(feature_scores[sorted_idx], inter_ind), color='b')
                else:
                    ax.barh(feature_names[sorted_idx], feature_scores[sorted_idx], color='b')
                ax.set_xlabel("Feature Importance")
                ax.set_ylabel("Feature")
                plt.grid()
                plt.tight_layout()
                fig.savefig(os.path.join(folder, f'{name}_{meth}.png'))
                plt.close(fig)
                break



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    num_distance_distr = 8
    distance_distr = {}
    n_samples = 5
    random_uniform_values = np.random.rand(n_samples)
    for i in range(num_distance_distr):
        distance_distr[i] = sample_empirical_dist(i, random_uniform_values)
    # Create the dataset and the labels
    # TODO: That would be interesting to compare volumes and count of vehicles
    df = input_data()
    # heatmap plot for MPR vs. Distr. (Volume=1800, TR=0.02)
    heatmap_data1 =  heat_map_plot(df, id_var1="Distr.", id_var2="MPR", res_var="LMTs",
                  id_var3="Volume", spec_val3=1800, id_var4="Truck-Ratio", spec_val4=0.02,
                  name="heat_map_", return_data=True)
    # heatmap plot for Volume vs. Distr. (MPR=0, TR=0.02)
    heatmap_data2 = heat_map_plot(df, id_var1="Distr.", id_var2="Volume", res_var="LMTs",
                  id_var3="MPR", spec_val3=0, id_var4="Truck-Ratio", spec_val4=0.02,
                  name="heat_map_", return_data=True)
    heat_map_two_subplot(heatmap_data1, heatmap_data2, ["Distr.", "Distr."], ["MPR", "Volume"], ["LMTs", "LMTs"],
                        "heatmap_both_plots")
    featVec = ['Intercept', 'Volume', 'MPR', 'Truck-Ratio']
    res_var = "LMTs"
    test_size = 0.2
    df = df[(df['Volume'] < 2000)]
    X_train, y_train, X_test, y_test = data_split(df, featVec, res_var, test_size)
    methods = ["Log Linear Regression", "Negative Binomial", "RF Regressor", "XG Boost"] # ["RF Regressor", "XG Boost", "Regression", "Poisson", "Negative Binomial"] #["XG Boost", "Mix", "Regression", "Poisson", "Negative Binomial"]
    res_heads = ["Methods"]
    for key in ["sqErr", "R2", "sampe", "rmsle"]:
        for setname in ["Tr", "Te"]:
            res_heads.append(key + setname)
    data_mix = pd.concat([y_train, X_train], axis=1)
    # model = fit_glm_with_family(y_train, X_train, 'mix', data_mix)
    train_col_header = [f"train {i}" for i in range(1, len(X_train)+1)]
    test_col_header = [f"test {i}" for i in range(1, len(X_test)+1)]
    train_col_headerRes = [f"trainRes {i}" for i in range(1, len(X_train) + 1)]
    test_col_headerRes = [f"testRes {i}" for i in range(1, len(X_test) + 1)]
    headers_dict_res = {"Tr": train_col_headerRes, "Te": test_col_headerRes}
    df_res = pd.DataFrame(columns=res_heads+train_col_header+test_col_header)
    fitted_models = []
    for meth in methods:
        trans = lambda y: (np.exp(y) if meth == "Log Linear Regression" else y)
        if meth == "Log Linear Regression":
            # model = fit_glm_with_family(y_train, X_train, 'gaussian', data_mix, group_var="X1")
            model = fit_regression(y_train, X_train)
        else:
            model = fit_glm_with_family(y_train, X_train, meth, data_mix, group_var="X1")
        fitted_models.append((meth, model))
        index = len(df_res)
        df_res.loc[index, "Methods"] = meth
        for setname, (X_samples, y_samples), header in zip(["Tr", "Te"], [(X_train, y_train), (X_test, y_test)],
                                                           [train_col_header, test_col_header]):
            y_pred = trans(model.predict(X_samples))
            y_pred = pd.DataFrame(y_pred, columns=y_samples.columns)
            y_pred = np.abs(y_pred)
            abs_error = np.abs(y_pred - y_samples)
            R2 = 1 - (np.sum(np.power(abs_error, 2)) / np.sum(np.power(np.average(y_samples) - y_pred, 2)))
            R2 = np.mean(R2)
            rel_errors = abs_error / y_samples
            mse_errors = np.sqrt(np.average((y_samples - y_pred) **2))
            sampe_i = abs_error / np.abs(y_pred + y_samples)
            sampe = np.average(sampe_i)
            rmsle_i = np.power(np.log(y_pred+1) - np.log(y_samples+1), 2)
            rmsle = np.sqrt(np.average(rmsle_i))
            df_res.loc[index, header] = sampe_i[y_pred.columns[0]].values
            df_res.loc[index, headers_dict_res[setname]] = (y_pred[y_pred.columns[0]].values - y_samples[y_pred.columns[0]].values) \
                                                           / y_samples[y_pred.columns[0]].values
            for key, value in zip(["sqErr", "R2", "sampe", "rmsle"], [mse_errors, R2, sampe, rmsle]):
                df_res.loc[index, key + setname] = value

    for name, header in zip(["train", "test"], [train_col_header, test_col_header]):
        df_res["mean_error_"+name] = df_res[header].mean(axis=1)
        df_res["std_error_"+name] = df_res[header].std(axis=1)
        merged_df = df_res[['Methods', *header]].melt(id_vars=['Methods'], value_vars=header,
                                                            value_name='error')
        bar_plot(data=merged_df, id_var="Methods", res_var="error", ylabel=r"$SMAPE_i$", name=name+"_box_plot_error")
    for setname in ["Tr", "Te"]:
        relative_error_histogram(df_res, id_var="Methods", val_var=headers_dict_res[setname], name=setname+"_relative_error")

    for err in ["sqErr", "R2", "sampe", "rmsle"]:
        header = []
        for mod in ["Te", "Tr"]:
            header.append(err+mod)
        merged_df = df_res[['Methods', *header]].melt(id_vars=['Methods'], value_vars=header, value_name=err)
        bar_plot(data=merged_df, id_var="Methods", res_var=err, ylabel=err, name=err + "_box_plot_error",
                 hu_var='variable', type="bar")

    feature_importance(fitted_models,
                       feature_names=featVec + [r"$X_" + str(i) + "$" for i in range(1, X_train.shape[1]-len(featVec)+1)],
                       name="FeatureImportance")
    for index, row in df_res.iterrows():
        print(f'{df_res["Methods"]}: Mean Err {df_res["mean_error_test"]}, Std Err {df_res["std_error_test"]}')
    df_res_r = df_res[["sqErrTe", "sampeTe", "rmsleTe", "R2Te"]]
    print(df_res_r)
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

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    xs = [0.6, 0.65, 1.65, 2.65, 3.65, 4.65, 5.65, 6.65, 8.0]
    ys_cars = [0.0, 0.1, 0.43, 0.76, 0.89, 0.96, 1.0, 1.0, 1.0]
    ys_trucks = [0.0, 0.0, 0.2, 0.27, 0.57, 0.77, 0.9, 1.0, 1.0]
    ax1.plot(xs, ys_cars, linewidth=2.5, c='#539ecd', linestyle='-', marker='o', mfc='black')
    for i_x, i_y in zip(xs[:-1], ys_cars[:-1]):
        ax1.text(i_x + 0.02, i_y, f'({i_x}, {i_y})')

    ax1.set_title('Passenger Cars')
    ax2.set_title('Trucks')
    ax1.set_xlim([0.55, 8.0])
    ax1.fill_between(xs, ys_cars, color='#539ecd')
    ax1.set_xlabel(r"CC1")
    ax1.set_ylabel(r"cumulative distribution")

    ax2.plot(xs[1:], ys_trucks[1:], linewidth=2.5, c='#539ecd', linestyle='-', marker='o', mfc='black')
    for i_x, i_y in zip(xs[1:-1], ys_trucks[1:-1]):
        ax2.text(i_x + 0.02, i_y, f'({i_x}, {i_y})')

    ax2.set_xlim([0.55, 8.0])
    ax2.fill_between(xs, ys_trucks, color='#539ecd')
    ax2.set_xlabel(r"CC1")
    ax2.set_xlabel(r"CC1")
    ax2.set_ylabel(r"cumulative distribution")

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
