# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.import warnings
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import numpy as np
import pandas as pd
from cycler import cycler

pd.options.mode.chained_assignment = None

import pdb
# Set Safety Reduction Factor (SRF)
SRF = 0.6
CMPR_SRF = 0.75
SRF_Dict = {0.6: "AV-06", 0.75: "AV-075", 0.7: "AV-Mix-06-075"}
SRF_Name = SRF_Dict[SRF] # "06"
CMPR_SRF_Name = {0.6: "AV-06", 0.75: "AV-075"}[CMPR_SRF]
# Set csv file
names = {"06": 'SRF_06\\', "04": 'SRF_04\\', "AV-06": 'AV_SRF_06\\', "AV-075":"AV_SRF_075\\",
         "AV-FAR-06": "AV_FAR_SRF_06\\", "AV-Mix-06-075": "AV_SRF_MIX_06_075\\"}
csv_names = {"06": 'csv_data_12.csv', "04": 'csv_data_17.csv', "AV-06": 'csv_data_06.csv',
             "AV-075": 'csv_data_075.csv', "AV-FAR-06": 'csv_data_4.csv', "AV-Mix-06-075": 'csv_data_5.csv'}

data_folder_name = os.path.join('Data', names[SRF_Name])
csv_file_name = os.path.join(data_folder_name, csv_names[SRF_Name])
# Set the folder to save the figures
folder = os.path.join("Figures", names[SRF_Name])
if not os.path.exists(folder):
    os.mkdir(folder)
# Load style file for plotting
custom_style_path = "science-ieee.mplstyle"
plt.style.use([custom_style_path])
DPI = 600
unit_conversion = {"Distr.": "Distr.", "Volume": "Volume (vph/2 lanes)", 'VehsDC(2)': "Flow Rate (vph)",
                   "LMTs": "LMTs", "TTCs": "TTCs", "Density": "Density (vpm)", 'Speed2': "Speed (mph)",
                   "Speed": "Speed (mph)", "Acceleration": r"Acceleration (ft/s$\,^2$)", "Delay": "Delay (spv)",
                   "Throughput": "Throughput (vphpl)"}

units = {"Volume": "(vph/2 lanes)"}

# Load the input data
def input_data(csv_file_name=csv_file_name):
    data = {
        "Scenario": [[1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4],
                     [5, 5, 5, 5, 5],
                     [6, 6, 6, 6, 6],
                     [7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8]],
        "volume": [[600, 800, 1500, 2500, 4000],
                   [600, 800, 1500, 2500, 4000],
                   [600, 800, 1500, 2500, 4000],
                   [600, 800, 1500, 2500, 4000],
                   [600, 800, 1500, 2500, 4000],
                   [600, 800, 1500, 2500, 4000],
                   [600, 800, 1500, 2500, 4000],
                   [600, 800, 1500, 2500, 4000],
                   ],
        "Delay": [[6.79, 8.07, 97.34, 861.28, 973.03],
                  [5.24, 7.59, 115.23, 869.41, 952.23],
                  [5.65, 7.56, 105.27, 881.73, 957.56],
                  [5.69, 7.60, 87.59, 915.80, 939.12],
                  [5.78, 9.00, 82.23, 837.65, 872.82],
                  [6.84, 10.89, 119.65, 868.86, 901.25],
                  [6.73, 10.87, 102.84, 852.61, 889.68],
                  [8.03, 12.09, 125.29, 854.30, 916.87]],
        "LMTs": [[24, 24, 504, 1056, 1092],
                 [24, 24, 468, 936, 1032],
                 [12, 24, 408, 864, 852],
                 [12, 24, 372, 852, 840],
                 [12, 0, 216, 672, 588],
                 [0, 0, 204, 456, 480],
                 [0, 12, 240, 480, 492],
                 [0, 0, 108, 336, 288]],
        "TTCs": [[24, 24, 504, 1056, 1092],
                 [24, 24, 468, 936, 1032],
                 [12, 24, 408, 864, 852],
                 [12, 24, 372, 852, 840],
                 [12, 0, 216, 672, 588],
                 [0, 0, 204, 456, 480],
                 [0, 12, 240, 480, 492],
                 [0, 0, 108, 336, 288]],
        "Throughput": [[838, 1066, 2146, 3586, 3660],
                       [835, 1066, 2170, 3595, 3655],
                       [835, 1066, 2150, 3617, 3658],
                       [835, 1066, 2141, 3636, 3650],
                       [835, 1066, 2136, 3578, 3605],
                       [838, 1066, 2167, 3605, 3631],
                       [835, 1066, 2158, 3598, 3622],
                       [838, 1070, 2167, 3598, 3636]]
    }
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
    data['MPR'] = data['MPR'].apply(lambda x: f'{int(x * 100)} %')
    data['Truck-Ratio'] = data['Truck-Ratio'].apply(lambda x: f'{int(x * 100)} %')
    return data


# 3D surface volume (density) versus scenario versus TTCs
def vol_scenarios_ttc():
    data = {
        "Scenario": [[1, 1, 1, 1],
                     [2, 2, 2, 2],
                     [3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5],
                     [6, 6, 6, 6],
                     [7, 7, 7, 7]],
        "volume": [[16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128]],
        "ttc": [[16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128]]

    }
    tri_d_surface(np.array(data["volume"]), np.array(data["Scenario"]), np.array(data["Delay"]),
                  "Volumes", "Compliance Dist.", "Delay",
                  data["volume"][0], [d[0] for d in data["Scenario"]], "3d_vol_scen_delay")


def tri_d_surface(x, y, z, x_label, y_label, z_label, name, type="scatter"):
    # mpl.use('TkAgg')
    # plt.style.use('fivethirtyeight')  # Using a specific style (You might have set a style before)
    # Define a 3-D graph named “ax”
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Render the graph
    # ax.plot_trisurf(x, y, z, cmap=plt.cm.Spectral_r)
    # triang = mtri.Triangulation(x.flatten(), y.flatten())
    # surface = ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), triangles=triang.triangles, cmap='viridis')
    # Create a custom colormap with more colors
    norm = mpl.colors.Normalize()
    z_min, z_max = 400, 800
    X, Y = np.meshgrid(x, y)
    Z = np.array(z)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(Z)
    if type == "scatter":
        surface = ax.scatter3D(X, Y, z)
    else:
        surface = ax.plot_surface(X, Y, Z, alpha=0.9, linewidth=0, cmap=plt.cm.viridis, antialiased=False,
                shade=False, rstride=1, cstride=1)
        ax.contour(X, Y, Z, levels=10, linestyles="solid", alpha=1.0, antialiased=True, colors="black")

        # surface = ax.plot_surface(X, Y, z, cmap=plt.cm.viridis, facecolors=mpl.cm.viridis(norm(z)), rstride=1, cstride=1, vmin=np.nanmin(z), vmax=np.nanmax(z))
    plt.colorbar(mappable, shrink=0.5, aspect=10, pad=0.1)
    # Add grid lines for xticks and yticks (in the XY plane)
    # Turn off the minor ticks for both x-axis and y-axis

    # Specify the axis ticks
    ax.set_xticks(x)
    ax.set_yticks(y)
    # ax.set_zticks(np.arange(1, 1.2, step=0.2))

    # Specify the title and axis titles
    # ax.set_title("Probability of Hypertension by Age and Weight")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(z_label, rotation=90)
    # rotates labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-15)

    # Specify the view angle of the graph
    ax.view_init(elev=20, azim=45)
    name = f"3D_plot_{name}.png".replace("%", "").replace(" ", "")
    # Save the graph
    plt.savefig(folder + name, dpi=DPI)
    # Close the plot
    plt.close(fig)


# 3D surface volume/(4 scenarios) versus delay OR throughput
def vol_vs_scenarios_delay(key="Delay") -> None:
    da = {
        "Scenario": [[1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4],
                     [5, 5, 5, 5, 5],
                     [6, 6, 6, 6, 6],
                     [7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8]],
        "volume0": [[600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    ],
        "Delay": [[6.79, 8.07, 97.34, 861.28, 973.03],
                  [5.24, 7.59, 115.23, 869.41, 952.23],
                  [5.65, 7.56, 105.27, 881.73, 957.56],
                  [5.69, 7.60, 87.59, 915.80, 939.12],
                  [5.78, 9.00, 82.23, 837.65, 872.82],
                  [6.84, 10.89, 119.65, 868.86, 901.25],
                  [6.73, 10.87, 102.84, 852.61, 889.68],
                  [8.03, 12.09, 125.29, 854.30, 916.87]],
        "LMTs": [[24, 24, 504, 1056, 1092],
                 [24, 24, 468, 936, 1032],
                 [12, 24, 408, 864, 852],
                 [12, 24, 372, 852, 840],
                 [12, 0, 216, 672, 588],
                 [0, 0, 204, 456, 480],
                 [0, 12, 240, 480, 492],
                 [0, 0, 108, 336, 288]],
        "TTCs": [[24, 24, 504, 1056, 1092],
                 [24, 24, 468, 936, 1032],
                 [12, 24, 408, 864, 852],
                 [12, 24, 372, 852, 840],
                 [12, 0, 216, 672, 588],
                 [0, 0, 204, 456, 480],
                 [0, 12, 240, 480, 492],
                 [0, 0, 108, 336, 288]],
        "Throughput": [[838, 1066, 2146, 3586, 3660],
                       [835, 1066, 2170, 3595, 3655],
                       [835, 1066, 2150, 3617, 3658],
                       [835, 1066, 2141, 3636, 3650],
                       [835, 1066, 2136, 3578, 3605],
                       [838, 1066, 2167, 3605, 3631],
                       [835, 1066, 2158, 3598, 3622],
                       [838, 1070, 2167, 3598, 3636]]
    }
    data = input_data()
    volume = data['volume'].unique()
    volume.sort()
    scenario = data['dist_distr'].unique()
    scenario.sort()
    delays = [[data[(data["dist_distr"] == dist) & (data["Volume"] == vol) &
                    (data["MPR"] == 0)]['Delay1'].values[0] for vol in volume]
              for dist in scenario]

    tri_d_surface(np.array(volume), np.array(scenario), delays,
                  "Volumes", "Compliance Dist.", "Delay", "3d_vol_scen_delay")

# 3D surface volume/(4 scenarios) versus delay OR throughput
def SRF_vs_scenarios_lmt(csv_file, id_var1='Distr.', id_var2='SRF', val_var1='Delay',
                 cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2=0.02, cri3='MPR', cri_val3=0,
                 name='3plot_distr_lmt_delay_mpr') -> None:
    da = {
        "Scenario": [[1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4],
                     [5, 5, 5, 5, 5],
                     [6, 6, 6, 6, 6],
                     [7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8]],
        "volume0": [[600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    [600, 800, 1500, 2500, 4000],
                    ],
        "Delay": [[6.79, 8.07, 97.34, 861.28, 973.03],
                  [5.24, 7.59, 115.23, 869.41, 952.23],
                  [5.65, 7.56, 105.27, 881.73, 957.56],
                  [5.69, 7.60, 87.59, 915.80, 939.12],
                  [5.78, 9.00, 82.23, 837.65, 872.82],
                  [6.84, 10.89, 119.65, 868.86, 901.25],
                  [6.73, 10.87, 102.84, 852.61, 889.68],
                  [8.03, 12.09, 125.29, 854.30, 916.87]],
        "LMTs": [[24, 24, 504, 1056, 1092],
                 [24, 24, 468, 936, 1032],
                 [12, 24, 408, 864, 852],
                 [12, 24, 372, 852, 840],
                 [12, 0, 216, 672, 588],
                 [0, 0, 204, 456, 480],
                 [0, 12, 240, 480, 492],
                 [0, 0, 108, 336, 288]],
        "TTCs": [[24, 24, 504, 1056, 1092],
                 [24, 24, 468, 936, 1032],
                 [12, 24, 408, 864, 852],
                 [12, 24, 372, 852, 840],
                 [12, 0, 216, 672, 588],
                 [0, 0, 204, 456, 480],
                 [0, 12, 240, 480, 492],
                 [0, 0, 108, 336, 288]],
        "Throughput": [[838, 1066, 2146, 3586, 3660],
                       [835, 1066, 2170, 3595, 3655],
                       [835, 1066, 2150, 3617, 3658],
                       [835, 1066, 2141, 3636, 3650],
                       [835, 1066, 2136, 3578, 3605],
                       [838, 1066, 2167, 3605, 3631],
                       [835, 1066, 2158, 3598, 3622],
                       [838, 1070, 2167, 3598, 3636]]
    }
    vol = 1800
    data1 = input_data()
    data2 = input_data(csv_file)
    data1['SRF'] = [SRF] * len(data1['link_behav'])
    data2['SRF'] = [CMPR_SRF] * len(data2['link_behav'])
    data = pd.concat([data1, data2])
    data.reset_index(drop=True, inplace=True)
    SRFs = data['SRF'].unique()
    SRFs.sort()
    scenario = data[id_var1].unique()
    scenario.sort()
    data = data[(data[cri1] == cri_val1) & (data[cri2] == cri_val2) & (data[cri3] == cri_val3)]
    res_val = [[data[(data[id_var1] == val_id_var1) & (data[id_var2] == srf)][val_var1].values[0] for srf in SRFs]
              for val_id_var1 in scenario]

    tri_d_surface(np.array(SRFs), np.array(scenario), res_val,
                  "SRF", "Compliance Dist.", val_var1, f'{name}_{cri1}_{cri_val1}_{cri2}_{cri_val2}.png',
                  type="3d-surface")


# 3D surface volume (density) versus scenario versus TTCs
def vol_scenarios_ttc():
    data = {
        "Scenario": [[1, 1, 1, 1],
                     [2, 2, 2, 2],
                     [3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5],
                     [6, 6, 6, 6],
                     [7, 7, 7, 7]],
        "volume": [[16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128],
                   [16, 32, 64, 128]],
        "ttc": [[16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128]]
    }
    data = input_data()
    volume = data['volume'].unique()
    volume.sort()
    scenario = data['dist_distr'].unique()
    scenario.sort()
    LMTs = [[data[(data["dist_distr"] == dist) & (data["Volume"] == vol) &
                  (data["MPR(0)"] == 0)]['VehsDC(1)'].values[0] for vol in volume]
            for dist in scenario]

    tri_d_surface(np.array(volume), np.array(scenario), LMTs,
                  "Volumes", "Compliance Dist.", "LMTs", "3d_vol_scen_lmts")


# 3D surface MP rate/(4 scenarios) versus TTC
# We need to fix a volume for this case
def mpr_scenarios_ttc():
    data = {
        "Scenario": [[4, 4, 4, 4],
                     [5, 5, 5, 5],
                     [6, 6, 6, 6],
                     [7, 7, 7, 7]],
        "MPR": [[16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128],
                [16, 32, 64, 128]],
        "Delay": [[16, 32, 64, 128],
                  [16, 32, 64, 128],
                  [16, 32, 64, 128],
                  [16, 32, 64, 128]],
        "Throughput": [200, 400, 800]

    }
    data = input_data()
    MPRs = data['MPR(0)'].unique()
    MPRs.sort()
    scenario = data['dist_distr'].unique()
    scenario.sort()
    LMTs = [[data[(data["dist_distr"] == dist) & (data["MPR(0)"] == mpr) &
                  (data["volume"] == 1500)]['VehsDC(1)'].values[0] for mpr in MPRs]
            for dist in scenario]

    tri_d_surface(np.array(MPRs), np.array(scenario), LMTs,
                  "MPR", "Compliance Dist.", "LMTs", "3d_mpr_scen_lmts")


def one_dim_box_plot_ttc():
    # compute this for each scenario
    data = {
        "volumes": [2000],
        "delays": [100],
        "throghput": [222]  # set it to the other axis
    }
    # compute this for each scenario
    data = {
        "volumes": [2000],
        "TTCs": [100],
    }
    data = {
        "volumes": [2000],
        "Number of late merges at the taper (LMT)": [100],
    }

    # time versus density for each volume
    data = {
        "Time": [2000],
        "Density": [100],
    }


def box_plot_ttc():
    # load test data
    exercise = sns.load_dataset("exercise")
    N = exercise["pulse"]
    data = {
        "Densities": list(exercise["time"].array),
        "TTCs": list(exercise["pulse"].array),
        "CV-MPR": list(exercise["kind"].array)
    }
    data = pd.DataFrame(data)
    data = input_data()
    # plot
    fig = plt.figure()
    sns.set(style='whitegrid', font_scale=1.15)
    #  g = sns.catplot(x="Densities", y="TTCs", hue="CV-MPR", data=data, kind="box")
    g = sns.catplot(x="Volume", y="LMTs", hue="MPR", data=data, kind="bar")
    # hatches must equal the number of hues (3 in this case)
    hatches = ['//', '..', 'xx', '\\']

    # iterate through each subplot / Facet
    for ax in g.axes.flat:
        # select the correct patches
        patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
        # the number of patches should be evenly divisible by the number of hatches
        h = hatches * (len(patches) // len(hatches))
        # iterate through the patches for each subplot
        for patch, hatch in zip(patches, h):
            patch.set_hatch(hatch)
            c = patch.get_facecolor()
            patch.set_edgecolor(c)
            patch.set_facecolor('none')

    plt.xlabel('Volume')
    plt.ylabel('LMTs')
    plt.savefig(folder + "vols_ttc_mpr.png")
    plt.close(fig)


def bar_plot_ttc(data=None, id_var1='Distr.', id_var2='MPR', val_var1='LMTs', val_var2='Delay',
                 cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2=0.02, name='barplot_distr_lmt_delay_mpr'):
    if data is None:
        data = input_data()
    # plot
    fig, ax = plt.subplots(1, 1)
    # Define a custom color cycle that includes orange
    custom_cycler = cycler(color=['b', 'r', 'g', 'orange'])

    current_cycler = plt.rcParams['axes.prop_cycle']
    updated_cycler = custom_cycler + cycler(linestyle=current_cycler.by_key()['linestyle'])

    # Set the updated cycler for the current axes
    plt.gca().set_prop_cycle(updated_cycler)

    # Apply the custom color cycle to the current axes
    plt.rcParams['axes.prop_cycle'] = custom_cycler

    # sns.set(style='whitegrid', font_scale=1.15)
    #  g = sns.catplot(x="Densities", y="TTCs", hue="CV-MPR", data=data, kind="box")
    df = data[(data[cri1] == cri_val1) & (data[cri2] == cri_val2)]
    # g = sns.catplot(x="Volume", y="LMTs", hue="MPR", data=df, kind="box")
    # Melt the DataFrame to prepare for sns.barplot
    # melted_df = pd.melt(df, id_vars=[id_var1, id_var2], value_vars=[val_var1, val_var2], var_name='Type',
    #                     value_name='Amount')
    # bottom_values = [0] * len(melted_df[melted_df['Type'] == val_var1])
    ylabel = unit_conversion[val_var1]
    if val_var2 is not None:
        ylabel += " / " + unit_conversion[val_var2]
        df[val_var2] += df[val_var1]
        df[val_var2] += 1.0  # offset
        sns.barplot(data=df, x=id_var1, y=val_var2, hue=id_var2, errorbar=None, ax=ax, alpha=0.7)
    sns.barplot(data=df, x=id_var1, y=val_var1, hue=id_var2, errorbar=None, hatch="xx", ax=ax, alpha=0.7)

    if id_var2 == "SRF" and val_var1 == 'Throughput':
        low, top = plt.ylim()  # return the current xlim
        plt.ylim((low + 1200, top))
        # Adjust the legend position

    # Show the legend only for the "Age" bars
    handles, labels = ax.get_legend_handles_labels()
    if val_var2 is not None:
        ax.legend(handles=handles[:len(handles) // 2], labels=labels[:len(handles) // 2], title=id_var2)
    ax.grid()
    plt.xlabel(unit_conversion[id_var1])
    plt.ylabel(ylabel)
    fig.savefig(os.path.join(folder, f'{name}_{cri1}_{cri_val1}_{cri2}_{cri_val2}.png'.replace('%', '').replace(' ', '')),
                bbox_inches="tight")
    plt.close(fig)
    plt.rcParams['axes.prop_cycle'] = current_cycler
    return


def bar_plot_srfs(csv_file, id_var1 = 'Distr.', id_var2 = 'SRF', val_var1 = 'LMTs', val_var2 = 'Delay',
                 cri1 = 'Volume', cri_val1 = 1800, cri2 = 'Truck-Ratio', cri_val2 = 0.02,
                 cri3 = "MPR", cri_val3=0.0, name = 'barplot_distr_lmt_delay_mpr'):
    data1 = input_data()
    data2 = input_data(csv_file)
    data1['SRF'] = [SRF] * len(data1['link_behav'])
    data2['SRF'] = [CMPR_SRF] * len(data2['link_behav'])
    data = pd.concat([data1, data2])
    data = data[data[cri3] == cri_val3]
    data.reset_index(drop=True, inplace=True)

    bar_plot_ttc(data, id_var1, id_var2, val_var1, val_var2,
                 cri1, cri_val1, cri2, cri_val2, name + cri3 + "_" + str(cri_val3))


def box_plot_speeds(csv_file_name, id_var1='Distr.', id_var2='Acceleration', cri1 = 'Volume', cri_val1 = 1800,
                    cri2 = 'Truck-Ratio', cri_val2 = 0.02, name="boxplot_distr_acceleration_mpr_0 %"):
    csv_file = os.path.join(data_folder_name, "speed", csv_file_name)
    data = pd.read_csv(csv_file)
    data['Distribution'] = data['Distribution'].apply(lambda x: 6 if x == 7 else (7 if x == 6 else x))
    data.rename(columns={'Distribution': 'Distr.', 'SPEED': 'Speed', 'ACCELERATION': 'Acceleration'}, inplace=True)
    if id_var2 == 'Acceleration':
        data = data[data[id_var2] > 0]
    custom_cycler = cycler(color=['b', 'r', 'g', 'orange'])
    current_cycler = plt.rcParams['axes.prop_cycle']
    updated_cycler = custom_cycler + cycler(linestyle=current_cycler.by_key()['linestyle'])
    # Set the updated cycler for the current axes
    plt.rcParams['axes.prop_cycle'] = updated_cycler
    # plot
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(x=id_var1, y=id_var2, hue=None, data=data, color='b')
    plt.xlabel(unit_conversion["Distr."])
    plt.ylabel(unit_conversion[id_var2])
    ax.minorticks_off()
    ax.grid()
    plt.tight_layout()
    fig.savefig(os.path.join(folder, f'{name}_{cri1}_{cri_val1}_{cri2}_{cri_val2}.png'
                             .replace('%', '').replace(' ', '')))
    plt.close(fig)
    plt.rcParams['axes.prop_cycle'] = current_cycler
    return



def plot_Compliance_dists() -> None:
    # some settings

    # sns.set_style("darkgrid")
    # sns.set()
    # fig = plt.figure(figsize=(15, 15))
    xs = [[50.0, 600.0, 600.0, 1200],
          [50.0, 600.0, 1200.0, 1200],
          [50.0, 150.0, 1000.0, 1400.0],
          [50.0, 460.0, 500.0, 900.0, 1000.0, 1390.0, 1500.0],
          [50.0, 600.0, 2500.0, 2500],
          [50.0, 300.0, 600.0, 1000.0, 1500.0, 2000.0, 2500.0],
          [50.0, 300.0, 600.0, 1000.0, 1500.0, 2000.0, 2500.0],
          [50.0, 460.0, 600.0, 850.0, 940.0, 980.0, 1000.0, 1380.0, 1440.0, 1470.0, 1510.0,
           1840.0, 1940.0, 1970.0, 2000.0, 2330.0, 2430.0, 2470.0, 2500.0]
          ]
    ys = [[0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 1.0],
          [0.0, 0.25, 0.25, 1.0],
          [0.0, 0.0, 0.30, 0.40, 0.60, 0.75, 1.0],
          [0.0, 0.0, 0.0, 1.0],
          [0.0, 0.0, 0.1, 0.1, 0.5, 0.7, 1.0],
          [0.0, 0.0, 0.1, 0.20, 0.38, 0.6, 1.0],
          [0.0, 0.0, 0.0, 0.02, 0.06, 0.13, 0.2, 0.22, 0.25, 0.3, 0.38, 0.43,
           0.47, 0.52, 0.60, 0.65, 0.77, 0.84, 1.0]
          ]

    xs = [[50.0, 600.0, 600.0, 1200],
          [50.0, 600.0, 600.0, 1200.0],
          [50.0, 600.00, 600.0, 730.0, 880.0, 1020.0, 1080.0, 1120.0, 1200.0],
          [50.0, 600.0, 600.0, 1000, 1200.0],
          [50.0, 1200.0, 1200.0, 2500],
          [50.0, 1200.0, 1200.0, 2500.0],
          [50.0, 800.0, 1200.0, 2500.0],
          [50.0, 1200.0, 1200.0, 2300.0, 2500.0]
          ]
    ys = [[0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.2, 1.0],
          [0.0, 0.0, 0.20, 0.22, 0.26, 0.39, 0.47, 0.60, 1.0],
          [0.0, 0.0, 0.2, 0.2, 1.0],
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.20, 1.0],
          [0.0, 0.0, 0.20, 1.0],
          [0.0, 0.0, 0.2, 0.2, 1.0]
          ]

    data = {
        "xs": xs,
        "ys": ys
    }
    data = pd.DataFrame(data)

    for i in range(len(xs)):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if i == 2:
            ax.scatter(np.array(data["xs"][i])[[0, 1, 2, 8]], 1 - np.array(data["ys"][i])[[0, 1, 2, 8]],
                       marker='o', color='b')
            ax.plot(np.array(data["xs"][i]), 1 - np.array(data["ys"][i]), color='b')
        else:
            ax.plot(data["xs"][i], 1- np.array(data["ys"][i]), marker='o', color='b')
        # ax1.set_xlim([0.55, 8.0])
        ax.set_xlabel(r"Distance to the Taper (ft)")
        if i in [0, 4]:
            ax.set_ylabel(r"Cumulative Distribution")
        default_xticks = ax.get_xticks()
        ax.set_xticks(list(default_xticks))
        ax.set_xlim(0, None)
        ax.grid()
        fig.tight_layout()
        fig.savefig(folder + "Compliance_" + str(i + 1) + ".png".replace('%', '').replace(' ', ''))
        plt.close(fig)


def plot_line_plots(id_var1="Distr.", id_var2="MPR", val_var1=None, val_var2=['0 %'],
                    id_var3='Truck-Ratio', val_var3=0.02, res_var="LMTs",
                    spec_var="Volume", spec_val=1800, name="1d_distr_LMTS_specific_vol_mpr") -> None:
    data = input_data()
    # if not val_var1:
    #    val_var1 = data[id_var1].unique()
    data = data.sort_values(id_var1)
    # val_var1.sort()
    # LMTs = [data[(data["Distr."] == dist) & (data["Volume"] == vol) &
    #                 (data["MPR"] == mpr)]['LMTs'].values[0] for dist in distributions]
    # ToDO: one side delay and one side throughput -- h-axis: distributions
    # TODO: One side LMTs and onde side TTcs
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for curr_val in val_var2:
        val_var1 = data[(data[spec_var] == spec_val) & (data[id_var2] == curr_val)
                        & (data[id_var3] == val_var3)][id_var1]
        res_val = data[(data[spec_var] == spec_val) & (data[id_var2] == curr_val)
                       & (data[id_var3] == val_var3) & (data[id_var1].isin(val_var1.values))][res_var]
        if len(res_val) == 0:
            continue
        ax.plot(val_var1, res_val, marker='o')
    ax.legend([id_var2 + ' ' + str(curr_val) + ' ' + units.get(id_var2, "") for curr_val in val_var2])
    # ax1.set_xlim([0.55, 8.0])
    ax.set_xlabel(unit_conversion[id_var1])
    ax.set_ylabel(unit_conversion[res_var])
    # default_xticks = ax1.get_xticks()
    if all(map(lambda x: int(x) == x, val_var1)):
        ax.set_xticks(val_var1)
    # Turn off minor ticks
    ax.minorticks_off()
    ax.grid()
    # ax.set_xlim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f'{name}_{spec_var}_{spec_val}_{id_var3}_{val_var3}.png'.replace('%', '').replace(' ', '')))
    plt.close(fig)


def plot_line_plots_srfs(csv_file, id_var1="Distr.", id_var2="MPR", val_var1=None, val_var2=['0 %'],
                    id_var3='Truck-Ratio', val_var3=0.02, res_var="LMTs",
                    spec_var="Volume", spec_val=1800, name="1d_distr_LMTS_specific_vol_mpr") -> None:
    data = input_data()
    # if not val_var1:
    #    val_var1 = data[id_var1].unique()
    data.sort_values(by=id_var1, inplace=True)
    # val_var1.sort()
    # LMTs = [data[(data["Distr."] == dist) & (data["Volume"] == vol) &
    #                 (data["MPR"] == mpr)]['LMTs'].values[0] for dist in distributions]
    # ToDO: one side delay and one side throughput -- h-axis: distributions
    # TODO: One side LMTs and onde side TTcs
    current_cycler = plt.rcParams['axes.prop_cycle'][:]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for curr_val in val_var2:
        val_var1 = data[(data[spec_var] == spec_val) & (data[id_var2] == curr_val)
                        & (data[id_var3] == val_var3)][id_var1]
        res_val = data[(data[spec_var] == spec_val) & (data[id_var2] == curr_val)
                       & (data[id_var3] == val_var3) & (data[id_var1].isin(val_var1.values))][res_var]
        if len(res_val) == 0:
            continue
        ax.plot(val_var1, res_val, marker='o', linestyle='solid')
    ax.legend([id_var2 + ' ' + str(curr_val) + ' ' + units.get(id_var2, "") for curr_val in val_var2])
    # Reset the color cycler again to the start of the cycle
    data = input_data(csv_file)
    data.sort_values(by=id_var1, inplace=True)
    plt.gca().set_prop_cycle(current_cycler)
    for curr_val in val_var2:
        val_var1 = data[(data[spec_var] == spec_val) & (data[id_var2] == curr_val)
                        & (data[id_var3] == val_var3)][id_var1]
        res_val = data[(data[spec_var] == spec_val) & (data[id_var2] == curr_val)
                       & (data[id_var3] == val_var3) & (data[id_var1].isin(val_var1.values))][res_var]
        if len(res_val) == 0:
            continue
        ax.plot(val_var1, res_val, marker='o', linestyle='dashed')

    # ax1.set_xlim([0.55, 8.0])
    ax.set_xlabel(unit_conversion[id_var1])
    ax.set_ylabel(unit_conversion[res_var])
    # default_xticks = ax1.get_xticks()
    if all(map(lambda x: int(x) == x, val_var1)):
        ax.set_xticks(val_var1)
    # Turn off minor ticks
    ax.minorticks_off()
    ax.grid()
    # ax.set_xlim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f'{name}_{spec_var}_{spec_val}_{id_var3}_{val_var3}.png'.replace('%', '').replace(' ', '')))
    plt.close(fig)



def plot_Wiedemann_CC1():
    # some settings
    # sns.set_style("darkgrid")
    # sns.set()
    # fig = plt.figure(figsize=(15, 15))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = [0.6, 0.65, 1.65, 2.65, 3.65, 4.65, 5.65, 6.65, 8.0]
    ys_cars = [0.0, 0.1, 0.43, 0.76, 0.89, 0.96, 1.0, 1.0, 1.0]
    ys_trucks = [0.0, 0.0, 0.2, 0.27, 0.57, 0.77, 0.9, 1.0, 1.0]

    data = {
        "xs": xs,
        "ys_cars": ys_cars,
        "ys_trucks": ys_trucks
    }
    data = pd.DataFrame(data)

    # ax1.plot(data["xs"], data["ys_cars"], linewidth=2.5, c='black', linestyle='-', marker='o', mfc='black')
    # ax1.plot(data["xs"][1:], data["ys_trucks"][1:], linewidth=2.5, c='black', linestyle='--', marker='o', mfc='black')
    ax.plot(data["xs"], data["ys_cars"], marker='o')
    ax.plot(data["xs"][1:], data["ys_trucks"][1:], marker='o')
    # for i_x, i_y in zip(xs[:-1], ys_cars[:-1]):
    #    ax1.text(i_x + 0.02, i_y, f'({i_x}, {i_y})')

    ax.set_xlim([0.55, 8.0])
    # ax1.fill_between(xs, ys_cars, color='#539ecd')
    # ax1.set_xlabel(r"CC1", fontsize=30)
    # ax1.set_ylabel(r"Cumulative Distribution", fontsize=30)
    # ax1.legend(["Cars", "Trucks"], fontsize=30)
    ax.set_xlabel(r"CC1 (s)")
    ax.set_ylabel(r"Cumulative Distribution")
    ax.legend(["Cars", "Trucks"])
    ax.grid()
    fig.tight_layout()
    fig.savefig(folder + "Wiedemann_CC1_car" + ".png")
    plt.close(fig)
    """
    fig = plt.figure(figsize=(15, 15))
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.plot(data["xs"][1:], data["ys_trucks"][1:], linewidth=2.5, c='black', linestyle='-', marker='o', mfc='black')
    for i_x, i_y in zip(xs[1:-1], ys_trucks[1:-1]):
        ax2.text(i_x + 0.02, i_y, f'({i_x}, {i_y})')

    ax2.set_xlim([0.55, 8.0])
    # ax2.fill_between(xs, ys_trucks, color='#539ecd')
    ax2.set_xlabel(r"CC1", fontsize=30)
    ax2.set_xlabel(r"CC1", fontsize=30)
    ax2.set_ylabel(r"cumulative distribution", fontsize=30)
    ax2.set_title('Trucks', fontsize=30, pad=5)

    fig.tight_layout()
    fig.savefig(folder + "Wiedemann_CC1_truck" + ".png")
    """


def ttc_plot_line_plots(id_var1="Distr.", id_var2="Volume", val_var2=[1800], res_var="TTCs",
                        SRFs=["04", "06"], file_name="1_5time_to_collison.csv",
                        name="1d_distr_TTCs_specific_vol_") -> None:

    data_folder_names = [os.path.join('Data', names[srf]) for srf in SRFs]
    skip_rows = 0
    legends = []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, fd_name in enumerate(data_folder_names):
        ttc_file_name = os.path.join(fd_name, 'ttc', file_name)
        df = pd.read_csv(ttc_file_name, skiprows=skip_rows)
        for val in val_var2:
            # Read the data
            df_filtered = df[df[id_var2] == val].sort_values(id_var1)
            val_var1 = df_filtered[id_var1]
            res_val = df_filtered[res_var]
            ax.plot(val_var1, res_val, marker='o')
            legends.append("SRF %s Volume %d" %(SRFs[i], val))


    ax.set_xlabel(id_var1)
    ax.set_ylabel(res_var)
    ax.legend(legends)
    # default_xticks = ax1.get_xticks()
    if all(map(lambda x: int(x) == x, val_var1)):
        ax.set_xticks(val_var1)
    # Turn off minor ticks
    ax.minorticks_off()
    ax.grid()
    # ax.set_xlim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f'{name}_{val}_{val:.2f}.png'.replace('%', '').replace(' ', '')))
    plt.close(fig)

def scatter_plot_with_corr(file_name="1_5time_to_collison.csv", name="LMT_vs_TTC_1_5", id_var1="Distr.",
                           id_var2="Volume", val_var2=[1000, 1200, 1500, 1800], res_var="TTCs"):
    # Sample data with a positive correlation
    # np.random.seed(42)
    # LMTs = np.random.rand(50)
    # TTCs = 2 * LMTs + np.random.rand(50) * 0.2
    # Set the file for TTCS
    ttc_file_name = os.path.join(data_folder_name, 'ttc', file_name)
    skip_rows = 0
    # Read the data
    df = pd.read_csv(ttc_file_name, skiprows=skip_rows)

    data = input_data()
    data = data[data.index < len(df)]

    df = df[(df["Volume"] <= 1800)]
    data = data[(data["Volume"] <= 1800)]

    TTCs = df["TTCs"]
    LMTs = data["LMTs"]
    # Perform linear regression to find the best-fit line
    coefficients = np.polyfit(LMTs, TTCs, 1)
    m, b = coefficients

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Create a scatter plot
    ax.scatter(LMTs, TTCs, c='b', marker='o', label='Data Points')

    # Plot the regression line
    ax.plot(LMTs, m * LMTs + b, c='r', label='Regression Line')
    print(f"m: {m} b: {b}")
    # ax.text(5, 5*(10+m), f'TTCs = {m:.2f} * LMTs + ({b:.2f})')

    # Add labels and title
    ax.set_xlabel('LMTs')
    ax.set_ylabel('TTCs')
    # Add a legend
    ax.legend()
    ax.grid()
    # Save the plot
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name + ".png").replace('%', '').replace(' ', ''))
    plt.close(fig)

    df['Distr.'] = df['Distr.'].apply(lambda x: 6 if x == 7 else (7 if x == 6 else x))
    df = df.sort_values(by='Distr.')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for curr_val in val_var2:
        val_var1 = df[(df[id_var2] == curr_val)][id_var1]
        res_val = df[(df[id_var2] == curr_val)][res_var]
        ax.plot(val_var1, res_val, marker='o', linestyle='solid')
    ax.legend([id_var2 + ' ' + str(curr_val) + ' ' + units[id_var2] for curr_val in val_var2])
    ax.set_xlabel(unit_conversion[id_var1])
    ax.set_ylabel(unit_conversion[res_var])
    if all(map(lambda x: int(x) == x, val_var1)):
        ax.set_xticks(val_var1)
    # Turn off minor ticks
    ax.minorticks_off()
    ax.grid()
    fig.tight_layout()
    fig.savefig(
    os.path.join(folder, f'{name}_{id_var1}_{id_var2}_{res_var}.png'.replace('%', '').replace(' ', '')))
    plt.close(fig)

def percent_improve(AValues, BValues):
    for a, b in zip(AValues, BValues):
        print(f'Percentage Improve: {(a-b) / a * 100: .2f}')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 3D diagram of delays versus compliance rate distributions (a, b, c, d) and volumes.
    # vol_vs_scenarios_delay()
    # vol_scenarios_ttc()
    # mpr_scenarios_ttc()
    # box_plot_ttc()
    # bar_plot_ttc()
    # plot_Compliance_dists()
    # Scatters plot TTCs vs. LMTs

    CMPR_SFR_CSV_FILE = os.path.join('Data', names[CMPR_SRF_Name], csv_names[CMPR_SRF_Name])

    bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='Throughput', val_var2=None,
                  cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3='0 %', name='barplotSrfs_distr_throughput_mpr')

    box_plot_speeds("tot_speeds_data.csv", id_var1='Distr.', id_var2='Acceleration', name="boxplot_distr_acceleration_mpr_0 %")
    box_plot_speeds("tot_speeds_data.csv", id_var1='Distr.', id_var2='Speed', name="boxplot_distr_speed_mpr_0 %")


    scatter_plot_with_corr(file_name="1_5time_to_collison.csv", name="LMT_vs_TTC_1_5")
    scatter_plot_with_corr(file_name="2_5time_to_collison.csv", name="LMT_vs_TTC_2_5")


    # Plot ids: (Density, Distr[1,2,3,4]), val: VehsDC(2), sp: MPR: 0.0, truck-ratio: 0.02
    plot_Wiedemann_CC1()
    plot_Compliance_dists()

    bar_plot_ttc(id_var1='Distr.', id_var2='MPR', val_var1='LMTs', val_var2='Delay',
                 cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                 name='barplot_distr_lmt_delay_mpr')

    CMPR_SFR_CSV_FILE = os.path.join('Data', names[CMPR_SRF_Name], csv_names[CMPR_SRF_Name])

    bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='LMTs', val_var2='Delay',
                  cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3='20 %', name='barplotSrfs_distr_lmt_delay_mpr')

    # Plot ids: (Distr., mpr), val: Speed2, sp: truck-ratio: 0.02
    for response in ['LMTs', 'Delay', 'Throughput', 'Speed2']:
        val_var2 = ['0 %', '20 %', '50 %', '80 %']
        if SRF == 0.7:
            val_var2 = ['20 %', '50 %', '80 %']
        plot_line_plots_srfs(CMPR_SFR_CSV_FILE, id_var1="Distr.", id_var2="MPR",
                    val_var1=None, val_var2=val_var2,
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var=response, spec_var="Volume", spec_val=1800,
                    name="1dSRFs_distr_"+ response + "_specific_vol_mpr")

    SRF_vs_scenarios_lmt(CMPR_SFR_CSV_FILE, id_var1='Volume', id_var2='SRF', val_var1='LMTs',
                         cri1='Distr.', cri_val1=1, cri2='Truck-Ratio', cri_val2='2 %', cri3='MPR', cri_val3='0 %',
                         name='3plot_srf_volume_lmt_')

    SRF_vs_scenarios_lmt(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='LMTs',
                         cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %', cri3='MPR', cri_val3='0 %',
                         name='3plot_srf_distr_lmt_')
    SRF_vs_scenarios_lmt(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='Delay',
                         cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %', cri3='MPR', cri_val3='0 %',
                         name='3plot_srf_distr_delay_')
    SRF_vs_scenarios_lmt(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='Throughput',
                         cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %', cri3='MPR', cri_val3='0 %',
                         name='3plot_srf_distr_throughput_')


    for mpr in ['0 %', '20 %', '50 %', '80 %']:
        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='LMTs', val_var2='Delay',
                  cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3=mpr, name='barplotSrfs_distr_lmt_delay_mpr')

        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='Throughput', val_var2="Speed2",
                  cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3=mpr, name='barplotSrfs_distr_throughput_speed_mpr')

    for mpr in ['0 %', '20 %', '50 %', '80 %']:
        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='LMTs', val_var2=None,
                  cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3=mpr, name='barplotSrfs_distr_lmt_mpr')

        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='Delay', val_var2=None,
                      cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                      cri3="MPR", cri_val3=mpr, name='barplotSrfs_distr_delay_mpr')

        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='Throughput', val_var2=None,
                  cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3=mpr, name='barplotSrfs_distr_throughput_mpr')

        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Distr.', id_var2='SRF', val_var1='Speed2', val_var2=None,
                      cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                      cri3="MPR", cri_val3=mpr, name='barplotSrfs_distr_speed_mpr')



    for mpr in ['0 %', '20 %', '50 %', '80 %']:
        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Volume', id_var2='SRF', val_var1='LMTs', val_var2='Delay',
                  cri1='Distr.', cri_val1=1, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3=mpr, name='barplotSrfs_volume_lmt_delay_mpr')

        bar_plot_srfs(CMPR_SFR_CSV_FILE, id_var1='Volume', id_var2='SRF', val_var1='Throughput', val_var2="Speed2",
                  cri1='Distr.', cri_val1=1, cri2='Truck-Ratio', cri_val2='2 %',
                  cri3="MPR", cri_val3=mpr, name='barplotSrfs_volume_throughput_speed_mpr')



    # Plot ids: (Volume., Distr[1-4]), val: Throughput, sp: MPR: 0.0
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[1, 2, 3, 4],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="Throughput", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_throughput_specific_mpr_distr_firsthalf")

    # Plot ids: (Volume., Distr[5-8]), val: Throughput, sp: MPR: 0.0
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[5, 6, 7, 8],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="Throughput", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_throughput_specific_mpr_distr_secondhalf")

    # Plot ids: (Volume., Distr[1-4]), val: Speed, sp: MPR: 0.0
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[1, 2, 3, 4],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="Speed2", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_speed_specific_mpr_distr_firsthalf")

    # Plot ids: (Volume., Distr[5-8]), val: Speed, sp: MPR: 0.0
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[5, 6, 7, 8],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="Speed2", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_speed_specific_mpr_distr_secondhalf")

    # Plot ids: (Distr., MPR), val: (LMTs, Delays), sp: (Vol: 1800, truck-ratio: 0.02),
    scatter_plot_with_corr(file_name="1_5time_to_collison.csv", name="LMT_vs_TTC_1_5")
    scatter_plot_with_corr(file_name="2_5time_to_collison.csv", name="LMT_vs_TTC_2_5")

    # Plot distributions versus TTCs
    # ttc_plot_line_plots()

    bar_plot_ttc(id_var1='Distr.', id_var2='MPR', val_var1='LMTs', val_var2='Delay',
                 cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                 name='barplot_distr_lmt_delay_mpr')

    # Plot ids: (Vol, MPR), val: (LMTs, Delays), sp: (Distr: 1, truck-ratio: 0.02),
    bar_plot_ttc(id_var1='Volume', id_var2='MPR', val_var1='LMTs', val_var2='Delay',
                 cri1='Distr.', cri_val1=1, cri2='Truck-Ratio', cri_val2='2 %',
                 name='barplot_volume_lmt_delay_mpr')
    for mpr in ['0 %', '20 %', '50 %', '80 %']:
        # Plot ids: (Distr., truck-ratios), val: (LMTs, Delays), sp: (Vol: 1800, mpr: 0.0),
        bar_plot_ttc(id_var1='Distr.', id_var2='Truck-Ratio', val_var1='LMTs', val_var2='Delay',
                 cri1='Volume', cri_val1=1800, cri2='MPR', cri_val2=mpr,
                 name='barplot_distr_lmt_delay_truck_ratio')

        # Plot ids: (Distr., truck-ratios), val: (LMTs, Delays), sp: (Vol: 1800, mpr: 0.0),
        bar_plot_ttc(id_var1='Volume', id_var2='Truck-Ratio', val_var1='LMTs', val_var2='Delay',
                 cri1='Distr.', cri_val1=1, cri2='MPR', cri_val2=mpr,
                 name='barplot_volume_lmt_delay_truck_ratio')

    # Plot ids: (Distr., mpr), val: (throughput, speed2), sp: (Vol: 1800, truck-ratio: 0.02),                 
    bar_plot_ttc(id_var1='Distr.', id_var2='MPR', val_var1='Throughput', val_var2='Speed2',
                 cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                 name='barplot_distr_speed_throughput_mpr')

    # Plot ids: (Volume, mpr), val: (throughput, speed2), sp: (Distr.: 1, truck-ratio: 0.02),                 
    bar_plot_ttc(id_var1='Volume', id_var2='MPR', val_var1='Throughput', val_var2='Speed2',
                 cri1='Distr.', cri_val1=1, cri2='Truck-Ratio', cri_val2='2 %',
                 name='barplot_volume_speed_throughput_mpr')
    
    # Plot ids: (Distr., mpr), val: (LMTs, Throughput), sp: (Vol: 1800, truck-ratio: 0.02),                 
    bar_plot_ttc(id_var1='Distr.', id_var2='MPR', val_var1='LMTs', val_var2='Throughput',
                 cri1='Volume', cri_val1=1800, cri2='Truck-Ratio', cri_val2='2 %',
                 name='barplot_distr_lmt_throughput_mpr')

    # Plot ids: (Vol., mpr), val: (LMTs, Speed2), sp: (Distr.: 1, truck-ratio: 0.02),                 
    bar_plot_ttc(id_var1='Volume', id_var2='MPR', val_var1='LMTs', val_var2='Speed2',
                 cri1='Distr.', cri_val1=1, cri2='Truck-Ratio', cri_val2='2 %',
                 name='barplot_volume_lmt_speed_mpr')

    for tr in ['2 %', '10 %', '20 %']:
        # Plot ids: (Distr., mpr), val: LMTs, sp: truck-ratio: 0.02
        plot_line_plots(id_var1="Distr.", id_var2="MPR",
                    val_var1=None, val_var2=['0 %', '20 %', '50 %', '80 %'],
                    id_var3 = 'Truck-Ratio', val_var3 = tr,
                    res_var="LMTs", spec_var="Volume", spec_val=1800,
                    name="1d_distr_lmt_specific_vol_mpr")

        # Plot ids: (Distr., mpr), val: Delay, sp: truck-ratio: 0.02
        plot_line_plots(id_var1="Distr.", id_var2="MPR",
                    val_var1=None, val_var2=['0 %', '20 %', '50 %', '80 %'],
                    id_var3 = 'Truck-Ratio', val_var3 = tr,
                    res_var="Delay", spec_var="Volume", spec_val=1800,
                    name="1d_distr_delay_specific_vol_mpr")

        # Plot ids: (Distr., mpr), val: Throughput, sp: truck-ratio: 0.02
        plot_line_plots(id_var1="Distr.", id_var2="MPR",
                    val_var1=None, val_var2=['0 %', '20 %', '50 %', '80 %'],
                    id_var3 = 'Truck-Ratio', val_var3 = tr,
                    res_var="Throughput", spec_var="Volume", spec_val=1800,
                    name="1d_distr_throughput_specific_vol_mpr")
                    
        # Plot ids: (Distr., mpr), val: Speed2, sp: truck-ratio: 0.02
        plot_line_plots(id_var1="Distr.", id_var2="MPR",
                    val_var1=None, val_var2=['0 %', '20 %', '50 %', '80 %'],
                    id_var3='Truck-Ratio', val_var3=tr,
                    res_var="Speed2", spec_var="Volume", spec_val=1800,
                    name="1d_distr_speed_specific_vol_mpr")
    
    # Plot ids: (Volume., Distr[1-4]), val: LMTs, sp: MPR: 0.0             
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[1, 2, 3, 4],
                    id_var3 = 'Truck-Ratio', val_var3 = '2 %',
                    res_var="LMTs", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_lmt_specific_mpr_distr_firsthalf")

    # Plot ids: (Volume., Distr[5-8]), val: LMTs, sp: MPR: 0.0             
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[5, 6, 7, 8],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="LMTs", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_lmt_specific_mpr_distr_secondhalf")


    # Plot ids: (Volume., Distr[1-4]), val: Delay, sp: MPR: 0.0
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[1, 2, 3, 4],
                    id_var3 = 'Truck-Ratio', val_var3 = '2 %',
                    res_var="Delay", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_delay_specific_mpr_distr_firsthalf")

    # Plot ids: (Volume., Distr[5-8]), val: Delay, sp: MPR: 0.0
    plot_line_plots(id_var1="Volume", id_var2="Distr.",
                    val_var1=None, val_var2=[5, 6, 7, 8],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="Delay", spec_var="MPR", spec_val='0 %',
                    name="1d_volume_delay_specific_mpr_distr_secondhalf")



    # Plot ids: (Distr., Volumes[1000-1800]), val: LMTs, sp: MPR: 0.0, truck-ratio: 0.02
    plot_line_plots(id_var1="Distr.", id_var2="Volume",
                    val_var1=None, val_var2=[1000, 1200, 1500, 1800],
                    id_var3 = 'Truck-Ratio', val_var3 = '2 %',
                    res_var="LMTs", spec_var="MPR", spec_val='0 %',
                    name="1d_distr_lmt_specific_mpr_vol")

    # Plot ids: (Distr., Volumes[1000-1800]), val: Delay, sp: MPR: 0.0, truck-ratio: 0.02
    plot_line_plots(id_var1="Distr.", id_var2="Volume",
                    val_var1=None, val_var2=[1000, 1200, 1500, 1800],
                    id_var3 = 'Truck-Ratio', val_var3 = '2 %',
                    res_var="Delay", spec_var="MPR", spec_val='0 %',
                    name="1d_distr_delay_specific_mpr_vol")


    for mpr in ['0 %', '20 %', '50 %', '80 %']:
        # Plot ids: (Distr, truck-ratios), val: LMTs, sp: MPR: 0.0
        plot_line_plots(id_var1="Distr.", id_var2="Truck-Ratio",
                    val_var1=None, val_var2=['2 %', '10 %', '20 %'],
                    id_var3='MPR', val_var3=mpr,
                    res_var="LMTs", spec_var="Volume", spec_val=1800,
                    name="1d_distr_lmt_specific_vol_truck_ratio")

        # Plot ids: (Distr, truck-ratios), val: Delay, sp: MPR: 0.0
        plot_line_plots(id_var1="Distr.", id_var2="Truck-Ratio",
                    val_var1=None, val_var2=['2 %', '10 %', '20 %'],
                    id_var3='MPR', val_var3=mpr,
                    res_var="Delay", spec_var="Volume", spec_val=1800,
                    name="1d_distr_delay_specific_vol_truck_ratio")
                    
        # Plot ids: (Distr, truck-ratios), val: Speed2, sp: MPR: 0.0, Vol: 1800
        plot_line_plots(id_var1="Distr.", id_var2="Truck-Ratio",
                    val_var1=None, val_var2=['2 %', '10 %', '20 %'],
                    id_var3='MPR', val_var3=mpr,
                    res_var="Speed2", spec_var="Volume", spec_val=1800,
                    name="1d_distr_speed_specific_vol_truck_ratio")

        # Plot ids: (Distr, truck-ratios), val: Speed2, sp: MPR: 0.0, Vol: 1800
        plot_line_plots(id_var1="Distr.", id_var2="Truck-Ratio",
                    val_var1=None, val_var2=['2 %', '10 %', '20 %'],
                    id_var3='MPR', val_var3=mpr,
                    res_var="Throughput", spec_var="Volume", spec_val=1800,
                    name="1d_distr_throughput_specific_vol_truck_ratio")
 
    # Plot ids: (LMTs, Distr), val: Throughput, sp: MPR: 0.0, truck-ratio: 0.02                 
    plot_line_plots(id_var1="LMTs", id_var2="Distr.",
                    val_var1=None, val_var2=[1, 2, 3, 4],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="Throughput", spec_var="MPR", spec_val='0 %',
                    name="1d_lmt_throughput_specific_mpr_distr_firsthalf")

    # Plot ids: (LMTs, Distr[1,2,3,4]), val: Speed2, sp: MPR: 0.0, truck-ratio: 0.02                 
    plot_line_plots(id_var1="LMTs", id_var2="Distr.",
                    val_var1=None, val_var2=[1, 2, 3, 4],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="Speed2", spec_var="MPR", spec_val='0 %',
                    name="1d_lmt_speed_specific_mpr_distr_firsthalf")

    # Plot ids: (Density, Distr[1,2,3,4]), val: VehsDC(2), sp: MPR: 0.0, truck-ratio: 0.02
    plot_line_plots(id_var1="Density", id_var2="Distr.",
                    val_var1=None, val_var2=[1, 2, 3, 4],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="VehsDC(2)", spec_var="MPR", spec_val='0 %',
                    name="1d_density_volume_specific_mpr_distr_firsthalf")

    # Plot ids: (LMTs, Distr[5,6,7,8]), val: VehsDC(2), sp: MPR: 0.0, truck-ratio: 0.02
    plot_line_plots(id_var1="Density", id_var2="Distr.",
                    val_var1=None, val_var2=[5, 6, 7, 8],
                    id_var3='Truck-Ratio', val_var3='2 %',
                    res_var="VehsDC(2)", spec_var="MPR", spec_val='0 %',
                    name="1d_density_volume_specific_mpr_distr_secondhalf")



    # plot_line_plots(MPRs=[0, 0.2, 0.5, 0.8])
    # Plot the compliance distributions
    # plot_Compliance_dists()

# vol_vs_scenarios_delay()

# 3D diagram of  TTCs versus compliance rate distributions (a, b, c, d) and volumes.
# vol_vs_scenarios_ttcs()

# 3D diagram of LMTs versus compliance rate distributions (a, b, c, d) and volumes.
# vol_vs_scenarios_lmts()

# 3D diagram of TTCs versus compliance rate distributions (a, b, c, d) and MPRs.
# mpr_scenarios_ttc()


# Scatter plot of LMTs vs. TTCs.
# scatter_plot_with_corr()


# TTCs versus compliance rate distributions.

# Delays versus compliance rate distributions.

# Histogram of LMTs versus heavy vehicle ratio.
# box_plot_ttc()

# Histogram of delays versus distributions.
# box_plot_ttc()

# Traffic demand versus throughput.

# TODO: queue length (open lane, closed lane), queue delay (open lane, closed lane), velocity in the open lane
# TODO: box-plot: h-axis: volumes, v-axis: TTCs-top-LMTs  over different CV-MPRs (0, 20, 50, 80)%
# TODO: LMT vs. Throughput, LMT v.s. SPEED
# TODO: barplot for h-axis: Volume, v-axis: LMTs, hu: Distr. [1, 4]


# scatter_plot_with_corr()
# mpr_scenarios_ttc()
# vol_vs_scenarios_delay()
# plot_Compliance_dists()
# plot_Compliance_dists()
# Interpretation:
# 1- In the first half of distributions for the higher rate of trucks, LMTs for the
# traffic with the lower heavy vehicle ratio rate is higher, while, for the second half,
# the traffic with higher heavy vehicle rate has lower LMTs.
# 2- For MPR=0.2 the throughput is higher than MPR=0.0 but the speed of the open lane close to
# the taper is lower. The reason is there is a queue in the closed lane and the cars in the queu
# can merge with the minimal delay
