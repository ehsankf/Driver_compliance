
from __future__ import print_function
import os
import numpy as np
# COM-Server
import win32com.client as com


## Connecting the COM Server => Open a new Vissim Window:
Vissim = com.gencache.EnsureDispatch("Vissim.Vissim") #
# Vissim = com.Dispatch("Vissim.Vissim") # once the cache has been generated, its faster to call Dispatch which also creates the connection to Vissim.
# If you have installed multiple Vissim Versions, you can open a specific Vissim version adding the version number
# Vissim = com.gencache.EnsureDispatch("Vissim.Vissim.10") # Vissim 10
# Vissim = com.gencache.EnsureDispatch("Vissim.Vissim.22") # Vissim 2022


### for advanced users, with this command you can get all Constants from PTV Vissim with this command (not required for the example)
##import sys
##Constants = sys.modules[sys.modules[Vissim.__module__].__package__].constants

Path_of_COM_Basic_Commands_network = 'C:\\Users\\Public\\Documents\\PTV Vision\\PTV Vissim 2023\\Examples Training\\COM\\Basic Commands\\'

## Load a Vissim Network:
Filename               = os.path.join(Path_of_COM_Basic_Commands_network, 'COM Basic Commands.inpx')
flag_read_additionally = False # you can read network(elements) additionally, in this case set "flag_read_additionally" to true
Vissim.LoadNet(Filename, flag_read_additionally)

## Load a Layout:
Filename = os.path.join(Path_of_COM_Basic_Commands_network, 'COM Basic Commands.layx')
Vissim.LoadLayout(Filename)


## ========================================================================
# Read and Set attributes
#==========================================================================
# Note: All of the following commands can also be executed during a
# simulation.

# Read Link Name:
Link_number = 1
Name_of_Link = Vissim.Net.Links.ItemByKey(Link_number).AttValue('Name')
Vissim.Log(16384, 'Name of Link(%d): %s' % (Link_number, Name_of_Link))
print('Name of Link(%d): %s' % (Link_number, Name_of_Link))

# Set Link Name:
new_Name_of_Link = 'New Link Name'
Vissim.Net.Links.ItemByKey(Link_number).SetAttValue('Name', new_Name_of_Link)

# Set a signal controller program:
SC_number = 1 # SC = SignalController
SignalController = Vissim.Net.SignalControllers.ItemByKey(SC_number)
new_signal_programm_number = 2
SignalController.SetAttValue('ProgNo', new_signal_programm_number)

# Set relative flow of a static vehicle route of a static vehicle routing decision:
SVRD_number         = 1 # SVRD = Static Vehicle Routing Decision
SVR_number          = 1 # SVR = Static Vehicle Route (of a specific Static Vehicle Routing Decision)
new_relativ_flow    = 0.6
Vissim.Net.VehicleRoutingDecisionsStatic.ItemByKey(SVRD_number).VehRoutSta.ItemByKey(SVR_number).SetAttValue('RelFlow(1)', new_relativ_flow)
# 'RelFlow(1)' means the first defined time interval; to access the third defined time interval: 'RelFlow(3)'

# Set vehicle input:
VI_number   = 1 # VI = Vehicle Input
new_volume  = 600 # vehicles per hour
Vissim.Net.VehicleInputs.ItemByKey(VI_number).SetAttValue('Volume(1)', new_volume)



Vissim.Net.Links.ItemByKey(10000).SetAttValue("LnChgDistDistrDef", True)
A = Vissim.Net.Links.ItemByKey(10000).LnChgDistDistrs.Attributes.GetAll()
Vissim.Net.VehicleInputs.ItemByKey(VI_number).SetAttValue('VehComp(1)', 1)
Vissim.Net.VehicleInputs.ItemByKey(VI_number).SetAttValue('Volume(1)', new_volume)

# run the simulations
# Activate QuickMode:
Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
Vissim.SuspendUpdateGUI()   # stop updating of the complete Vissim workspace (network editor, list, chart and signal time table windows)
# Alternatively, load a layout (*.layx) where dynamic elements (vehicles and pedestrians) are not visible:
# Vissim.LoadLayout(os.path.join(Path_of_COM_Basic_Commands_network, 'COM Basic Commands - Hide vehicles.layx')) # loading a layout where vehicles are not displayed
End_of_simulation = 600
Vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation)
Sim_break_at = 0 # simulation second [s] => 0 means no break!
Vissim.Simulation.SetAttValue('SimBreakAt', Sim_break_at)
# Set maximum speed:
Vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)

for cnt_Sim in range(3):
    Vissim.Simulation.SetAttValue('RandSeed', cnt_Sim + 1) # Note: RandSeed 0 is not allowed
    Vissim.Simulation.RunContinuous()


## ========================================================================
# Saving
#==========================================================================
Filename = os.path.join(Path_of_COM_Basic_Commands_network, 'COM Basic Commands saved.inpx')
Vissim.SaveNetAs(Filename)
Filename = os.path.join(Path_of_COM_Basic_Commands_network, 'COM Basic Commands saved.layx')
Vissim.SaveLayout(Filename)

Vissim.Net.DrivingBehaviors.ItemByKey(1).AttValue("SafDistFactLnChg")
Vissim.Net.DrivingBehaviors.ItemByKey(1).AttValue('Name')
Vissim.Net.Links.ItemByKey(Link_number).AttValue("LinkBehavType")
Vissim.Net.Links.ItemByKey(Link_number).SetAttValue("LinkBehavType", 2)


# Add vehicle composition
# Vehicle Compositions
# Vissim.Net.VehicleCompositions.AddVehicleComposition(0, []) # unsigned int Key, SAFEARRAY(VARIANT) VehCompRelFlows
# Vissim.Net.VehicleCompositions.AddVehicleComposition(0, [Vissim.Net.VehicleTypes.ItemByKey(100), Vissim.Net.DesSpeedDistributions.ItemByKey(40)]) # unsigned int Key, SAFEARRAY(VARIANT) VehCompRelFlows
# Vissim.Net.VehicleCompositions.AddVehicleComposition(9, [Vissim.Net.VehicleTypes.ItemByKey(100), Vissim.Net.DesSpeedDistributions.ItemByKey(40), Vissim.Net.VehicleTypes.ItemByKey(200), Vissim.Net.DesSpeedDistributions.ItemByKey(30)]) # unsigned int Key, SAFEARRAY(VARIANT) VehCompRelFlows
# Vissim.Net.VehicleCompositions.ItemByKey(9).VehCompRelFlows.AddVehicleCompositionRelativeFlow(Vissim.Net.VehicleTypes.ItemByKey(300), Vissim.Net.DesSpeedDistributions.ItemByKey(25)) # IVehicleType* VehType, IDesSpeedDistribution* DesSpeedDistr

# Distribution number
# Vissim.Net.DistanceDistributions.ItemByKey

# Data measurement points and interval times for vehicles

## ========================================================================
# End Vissim
#==========================================================================
Vissim = None
