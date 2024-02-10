
from __future__ import print_function
import os
import pdb

import numpy as np
import pandas as pd
# COM-Server
import win32com.client as com

import subprocess
import sys
import pythoncom
import time

def restart_program():
    subprocess.Popen([sys.executable] + sys.argv, shell=True)
    sys.exit()

start_time = time.time()
# Your input
# Vehicle input number
VI_number = 1
# Link behaviors (safety reduction factor)
LinkDrivBehaves = [7] #[7, 8] # [7, 8, 9, 10] # [7: WZ, 8: WZ-AV-SRF-0.40, 9: AV-normal, 10: AV-Aggressive)
# Market penetration rate
MPRs = [(0, 0.01), (0.19, 0.01), (0.49, 0.01), (0.79, 0.01)]
# Heavy vehicle ratios
RelativeTruckFlows = [0.02] # [0.02, 0.1, 0.2] # [(cars, HGVs, AV cars, AV HGVs)]
# Distance distribution (compliance distribution)
DistDistr = [1, 2, 3, 4, 5, 6, 7, 8]
# Input volumes
Volumes = [800, 1000, 1200, 1500, 1800, 2000]
# Number of seeds per run
num_seeds = 1
# The base seed number
base_seed = 42
End_of_simulation = 600
# Path of network
Path_of_COM_Basic_Commands_network = 'E:\\E_Derive_COM Basic Commands\\'
# network file name
File_Name = 'AV_Lane_Change_Compliance_Ablation'
# csv file name to save the results
CSV_File_Name = 'csv_data.csv'

# Your code here

# Save csv file
def save_csv_with_incremental_name(filename, separator='_'):
    base_name, ext = os.path.splitext(filename)
    new_filename = filename

    counter = 1
    while os.path.exists(new_filename):
        new_filename = f"{base_name}{separator}{counter}{ext}"
        counter += 1

    df.to_csv(new_filename, mode='a', header=True, index=False)

# Get the attributes from csv file
def get_attributes(measure, attributes, num_seeds, count):
    attribute_values = dict()
    for j in range(num_seeds):
        idx = str(count - j)
        _attributes = [attr+f'({idx}, Total, All)' for attr in attributes]
        for i, attr in enumerate(attributes):
            if not attr in attribute_values:
                attribute_values[attr] = []
            value = measure.AttValue(_attributes[i])
            attribute_values[attr].append(value if not isinstance(value, type(None)) else 0)
    for attr in attributes:
        attribute_values[attr] = np.average(attribute_values[attr])
    return attribute_values
    # volume = Vissim.Net.Links.ItemByKey(Link_number).AttValue('Concatenate:LinkEvalSegs\Volume(Avg, Avg, All)')

# Print the data collection attributes Vehs, Speed, Acceleration, Length
def print_DC(DC_measurement_number, run_num):
    print("==================================Run: %d====================================="% (run_num))
    DC_measurement_ = Vissim.Net.DataCollectionMeasurements.ItemByKey(DC_measurement_number)
    No_Veh = DC_measurement_.AttValue('Vehs        (Current,Total,All)')  # number of vehicles
    Speed = DC_measurement_.AttValue('Speed       (Current,Total,All)')  # Speed of vehicles
    Acceleration = DC_measurement_.AttValue('Acceleration(Current,Total,All)')  # Acceleration of vehicles
    Length = DC_measurement_.AttValue('Length      (Current,Total,All)')  # Length of vehicles
    Vissim.Log(16384,
               'Data Collection #%d: Average values of all Simulations runs of 1st time intervall of all vehicle classes: ' % (
                   DC_measurement_number))
    print(
        'Data Collection #%d: Average values of all Simulations runs of 1st time intervall of all vehicle classes: ' % (
            DC_measurement_number))
    Vissim.Log(16384,
               '#vehicles: %d; Speed: %.2f; Acceleration: %.2f; Length: %.2f' % (No_Veh, Speed, Acceleration, Length))
    print('#vehicles: %d; Speed: %.2f; Acceleration: %.2f; Length: %.2f' % (No_Veh, Speed, Acceleration, Length))
    print("===========================================================================")
    return No_Veh, Speed, Acceleration

## Connecting the COM Server => Open a new Vissim Window:
# Vissim = com.gencache.EnsureDispatch("Vissim.Vissim") #
Vissim = com.Dispatch("Vissim.Vissim") # once the cache has been generated, its faster to call Dispatch which also creates the connection to Vissim.

### for advanced users, with this command you can get all Constants from PTV Vissim with this command (not required for the example)
##import sys
##Constants = sys.modules[sys.modules[Vissim.__module__].__package__].constants

# Create the file path for the network
file_path = os.path.join(Path_of_COM_Basic_Commands_network, CSV_File_Name)

## Load a Vissim Network:
Filename               = os.path.join(Path_of_COM_Basic_Commands_network, File_Name + '.inpx')
flag_read_additionally = False # you can read network(elements) additionally, in this case set "flag_read_additionally" to true
Vissim.LoadNet(Filename, flag_read_additionally)

## Load a Layout:
Filename = os.path.join(Path_of_COM_Basic_Commands_network, File_Name + '.layx')
Vissim.LoadLayout(Filename)


## ========================================================================
# Read and Set attributes
#==========================================================================
# Note: All of the following commands can also be executed during a
# simulation.

# Delete all previous simulation runs first:
for simRun in Vissim.Net.SimulationRuns:
    Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)

# run the simulations
# Activate QuickMode:
Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
Vissim.SuspendUpdateGUI()   # stop updating of the complete Vissim workspace (network editor, list, chart and signal time table windows)
# Alternatively, load a layout (*.layx) where dynamic elements (vehicles and pedestrians) are not visible:
# Vissim.LoadLayout(os.path.join(Path_of_COM_Basic_Commands_network, 'COM Basic Commands - Hide vehicles.layx')) # loading a layout where vehicles are not displayed
Vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation)
Sim_break_at = 0 # simulation second [s] => 0 means no break!
Vissim.Simulation.SetAttValue('SimBreakAt', Sim_break_at)
# Set maximum speed:
Vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)

# Connector number
Connector_number = 10000
# Data collection point
DC_measurement_number = 2
# Create an empty DataFrame with columns
columns = ["link_behav", "MPR(0)", "MPR(1)", "Car(0)", "Car(1)", "dist_distr", "volume", "num_seed"]
# Get the Data Collection Points
DC_measurement = [Vissim.Net.DataCollectionMeasurements.ItemByKey(nmbr) for nmbr in range(1, DC_measurement_number+1)]
DC_attributes = ('Vehs', 'Speed', 'Acceleration', 'QueueDelay')
columns.extend(attr + 'DC(' +  str(nmbr) + ')' for nmbr in range(1, DC_measurement_number+1)
               for attr in DC_attributes)

# Get the results of Vehicle Travel Time Measurements:
Veh_TT_measurement_number = 1
Veh_TT_measurement = Vissim.Net.VehicleTravelTimeMeasurements.ItemByKey(Veh_TT_measurement_number)
Veh_Delay_measurement = Vissim.Net.DelayMeasurements.ItemByKey(Veh_TT_measurement_number)
Delay_attributes = ('Vehs', 'VehDelay')
columns.extend(attr + 'Delay(' +  str(Veh_TT_measurement_number) + ')' for attr in Delay_attributes)
# Define dataframe for saving csv files
df = pd.DataFrame(columns=columns)
# Loop over the link ehaviors
for link_behav in LinkDrivBehaves:
    # Get all the Links
    Link_count = Vissim.Net.Links.GetMultipleAttributes(["No"])
    # Set the behavior of the link to link_behav number
    for Link_number in Link_count:
        Vissim.Net.Links.ItemByKey(Link_number[0]).SetAttValue("LinkBehavType", link_behav)
    # Get the name of link
    link_behav_text = str(link_behav) + ' : ' + Vissim.Net.Links.ItemByKey(1).LinkBehavType.AttValue("Name")
    # Loop over Market Penetration Rates
    for id_mpr, mpr in enumerate(MPRs):
        Veh_composition_number = 1 if mpr[0] == 0 else 2
        Vissim.Net.VehicleInputs.ItemByKey(VI_number).SetAttValue("VehComp(1)", Veh_composition_number) # Assign vehicle composition to the input vehciles
        # Changing the relative flows of vehicle composition
        Rel_Flows = Vissim.Net.VehicleCompositions.ItemByKey(Veh_composition_number).VehCompRelFlows.GetAll()
        # Rel_Flows[0].SetAttValue('DesSpeedDistr', 50)  # Changing the desired speed distribution
        if Veh_composition_number == 2:
            Rel_Flows[1].SetAttValue('RelFlow', mpr[0])  # Changing the MPR for cars
            Rel_Flows[3].SetAttValue('RelFlow', mpr[1])
        # Loop over truck ratios
        for rel_flow in RelativeTruckFlows:
            rel_flow_idx = 2 if Veh_composition_number == 2 else 1
            rel_flow_car = 1-rel_flow-mpr[0]-(mpr[1] if mpr[0] != 0 else 0)
            Rel_Flows[0].SetAttValue('RelFlow', rel_flow_car)  # Changing the relative flow
            Rel_Flows[rel_flow_idx].SetAttValue('RelFlow', rel_flow)  # Changing the relative flow of the 2nd Relative Flow.
            # Vissim.Net.VehicleInputs.ItemByKey(VI_number).SetAttValue('VehComp(1)', veh_compos)
            # Distance distribution for lane change: 1-8
            for dis_distr in DistDistr:
                Vissim.Net.Links.ItemByKey(Connector_number).SetAttValue("LnChgDistDistrDef", dis_distr)
                # Different input volumes [800, 1000, 1500, 2000]
                for volume in Volumes:
                    print(f'Volume: {volume}')
                    Vissim.Net.VehicleInputs.ItemByKey(VI_number).SetAttValue('Volume(1)', volume)
                    # Different seeds # base_seed -- base_seed + num_seeds///
                    for cnt_Sim in range(base_seed, base_seed+num_seeds):
                        Vissim.Simulation.SetAttValue('RandSeed', cnt_Sim + 1)  # Note: RandSeed 0 is not allowed
                        Vissim.Simulation.RunContinuous()
                        # ["link_behav", "MPR(0)", "MPR(1)", "Car(0)", "Car(1)", "dist_distr", "volume", "num_seed"]
                        row = [link_behav_text, mpr[0], mpr[1], rel_flow_car, rel_flow, dis_distr, volume, cnt_Sim + 1]
                    # TODO: do we need to resume Graphics
                    # Vissim.ResumeUpdateGUI(True)  # allow updating of the complete Vissim workspace (network editor, list, chart and signal time table windows)
                    Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 0)  # deactivate QuickMode
                    for nmbr in range(DC_measurement_number):
                        DC_data = get_attributes(DC_measurement[nmbr], DC_attributes, num_seeds, Vissim.Net.SimulationRuns.Count)
                        row.extend([DC_data[key] for key in DC_attributes])
                    # DC_data = print_DC(DC_measurement_number, cnt_Sim + 1)
                    Delay_data = get_attributes(Veh_Delay_measurement, Delay_attributes, num_seeds, Vissim.Net.SimulationRuns.Count)
                    row.extend([Delay_data[key] for key in Delay_attributes])
                    df.loc[len(df)] = row  # Add a new row to the DataFrame
                    Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)  # activate QuickMode
                    # Vissim.SuspendUpdateGUI()

end_time = time.time()
print(f'Elapsed time {end_time - start_time} seconds')
Vissim.ResumeUpdateGUI(True)  # allow updating of the complete Vissim workspace (network editor, list, chart and signal time table windows)
Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 0)  # deactivate QuickMode
# Vissim.LoadLayout(os.path.join(Path_of_COM_Basic_Commands_network, 'COM Basic Commands.layx')) # loading a layout to display vehicles again
# df.to_csv(CSV_File_Name, mode='a', header=True, index=False)
save_csv_with_incremental_name(CSV_File_Name)

## ========================================================================
# Saving
#==========================================================================
Filename = os.path.join(Path_of_COM_Basic_Commands_network, f'COM Basic Commands saved_{Veh_composition_number}_04.inpx')
Vissim.SaveNetAs(Filename)
Filename = os.path.join(Path_of_COM_Basic_Commands_network, f'COM Basic Commands saved_{Veh_composition_number}_04.layx')
Vissim.SaveLayout(Filename)

## ========================================================================
# End Vissim
#==========================================================================
Vissim = None
pythoncom.CoUninitialize()
