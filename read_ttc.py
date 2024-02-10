
from __future__ import print_function

import os.path

import numpy as np
import pandas as pd

# Vehicle input number
VI_number = 1
# Link behaviors (safety reduction factor)
LinkDrivBehaves = [8] #[7, 8] # [7, 8, 9, 10] # [7: WZ, 8: WZ-AV-SRF-0.40, 9: AV-normal, 10: AV-Aggressive)
# Market penetration rate
MPRs = [(0, 0.01)] # [(0, 0.01), (0.19, 0.01), (0.49, 0.01), (0.79, 0.01)]
# Heavy vehicle ratios
RelativeTruckFlows = [0.02] # [0.02, 0.1, 0.2] # [(cars, HGVs, AV cars, AV HGVs)]
# Distance distribution (compliance distribution)
DistDistr = [1, 2, 3, 4, 5, 6, 7, 8]
# Input volumes
Volumes = [600, 800, 1000, 1200, 1500, 1800, 2000]
# Number of seeds per run
num_seeds = 5

columns = ["MPR", "truck", "Distr.", "Volume", "TTCs"]
folder_name = "SRF_04/ttc/"
csv_file_name = os.path.join(folder_name, "2_5ttc.csv")
new_filename = os.path.join(folder_name, "2_5time_to_collison.csv")
skip_rows = 0
data = pd.read_csv(csv_file_name, skiprows=skip_rows)
# Define dataframe for saving csv files
df = pd.DataFrame(columns=columns)
# Loop over the link ehaviors
index, count = 0, 1
for link_behav in LinkDrivBehaves:
    # Loop over Market Penetration Rates
    for id_mpr, mpr in enumerate(MPRs):
        # Loop over truck ratios
        for rel_flow in RelativeTruckFlows:
            # Loop over distributions [1, 2, 3, 4, 5, 6, 7, 8]
            rel_flow_car = 1 - rel_flow - mpr[0] - (mpr[1] if mpr[0] != 0 else 0)
            if rel_flow_car < 0.001:
                continue
            for dis_distr in DistDistr:
                # Different input volumes [600, 800, 1000, 1500, 2000]
                for volume in Volumes:
                    ttcs = 0
                    for cnt_Sim in range(num_seeds):
                        if index < len(data) and count == int(data.loc[index, "Summary Group "].split("Ablation_")[1].split('.')[0]):
                            ttcs += float(data.loc[index, " Total"])
                            index += 1
                        count += 1
                    time_to_collision = ttcs / 5.
                    print(f'count{count:d} MPR {mpr[0]:.2f} truck {rel_flow:.2f} Distr. {dis_distr:.2f}'
                          f' Volume {volume:.2f} TTCs {time_to_collision}')
                    df.loc[len(df)] =[mpr[0], rel_flow, dis_distr, volume, time_to_collision]

df.to_csv(new_filename, header=True, index=False)




