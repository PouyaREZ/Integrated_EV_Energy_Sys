from __future__ import division

#-------------------------------------------------------------------------------
# Name:        SolarPanels
# Purpose:     File full of solar panel simulations for use in urban
#              optimization code.
#
# Author:      Rob Best
# Modifier:    Pouya Rezazadeh
#
# Created:     23/07/2015
# Modified:    06/01/2019
# Copyright:   (c) Rob Best 2015
#-------------------------------------------------------------------------------

import numpy as np
#import math as mp
#import random as rp
#import sys
#import copy

# Year = 2020


####### AUXILIARY FUNCTION START ##########
def Computer(Efficiency, Panel_Size, Perez_Bins, Extraterrestrial_Irradiance, Kappa, n_Glass, T_ref, Pdc_0, Gamma, Adsorption_Coefficient, Soiling_Loss, Shading_Loss, Snow_Loss, Mismatch_Loss, Wiring_Loss, Connections_Loss, Light_Induced_Degradation, Nameplate_Rating, Age, Availability, Inverter_Eta_Ref, Inverter_Eta_Nom, Inverter_Pac_0, U_0, U_1, Solar_Cost_SFH, Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude):
    Day = np.floor(Hour/24)+1
#    Month_Days = [32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
#    for i in range(len(Month_Days)):
#        if Day < Month_Days[i]:
#            Month = i + 1
#            break

    # Convert inputs to radians
    Latitude = Latitude*np.pi/180
    Tilt = Tilt*np.pi/180
    Azimuth = Azimuth*np.pi/180

	# Calculate the Julian Date
#    a = np.floor((14-Month)/12)
#    y = Year+4800-a
#    m = Month+12*a-3
#    h = Hour-np.floor(Hour/24)*24
#    Julian_Day = Day+np.floor((153*m+2)/5)+365*y+np.floor(y/4)-np.floor(y/100)+np.floor(y/400)-32045+(h-12)/24+30/1440
#    n_star = Julian_Day-2451545.0009-Longitude/360
#    n = np.floor(n_star+0.5)

    # Calculate observer angles
#    Solar_Noon = 2451545.0009+Longitude/360+n
#    Solar_Mean_Anomaly = (357.5291+0.98560028*(Solar_Noon-2451545))-np.floor((357.5291+0.98560028*(Solar_Noon-2451545))/360)*360
#    Equation_of_Center = 1.9148*np.sin(Solar_Mean_Anomaly*np.pi/180)+0.0200*(2*Solar_Mean_Anomaly*np.pi/180)+0.0003*np.sin(3*Solar_Mean_Anomaly*np.pi/180)
#    Ecliptic_Longitude = (Solar_Mean_Anomaly+102.9372+Equation_of_Center+180)-np.floor((Solar_Mean_Anomaly+102.9372+Equation_of_Center+180)/360)*360
#    Solar_Transit = Solar_Noon+0.0053*np.sin(Solar_Mean_Anomaly*np.pi/180)-0.0069*np.sin(2*Ecliptic_Longitude*np.pi/180)
#    Declination = np.arcsin(np.sin(Ecliptic_Longitude*np.pi/180)*np.sin(23.45*np.pi/180))
    Declination = np.pi/180*23.45*np.sin(360/365*(2854+Day)*np.pi/180)           # Angle of Declination of the sun, radians
    Longitude_Time = 4*(UTC*15-Longitude)                                       # Correction of time for longitude, minutes
    B_Factor = (Day-1)*360/365                                                  # B is needed for Equation of Time, radians
    Equation_of_Time = 229.2*(0.000075)+229.2*(0.001868*np.cos(B_Factor)-0.032077*np.sin(B_Factor))-229.2*(0.014615*np.cos(2*B_Factor)+0.04089*np.sin(2*B_Factor))  # Equation of Time, minutes
    Time_Correction = Longitude_Time + Equation_of_Time                         # minutes
    Solar_Time = Hour+0.5+Time_Correction/60                                    # hours (i.e., 2:30 pm is 14.5)
    Hour_Angle = np.pi/180*15*(Solar_Time-12)                                   # radians
#    Hour_Angle = np.arccos((np.sin(-0.83*np.pi/180)-np.sin(Latitude)*np.sin(Declination))/(np.cos(Latitude)*np.cos(Declination)))

    # Calculate solar earth angles
    Solar_Zenith = np.arccos(np.sin(Latitude)*np.sin(Declination)+np.cos(Latitude)*(np.cos(Declination)*np.cos(Hour_Angle)))             # radians
#    if Hour_Angle > 0:
#        Solar_Azimuth = np.arccos((np.cos(Solar_Zenith)*np.sin(Latitude)-np.sin(Declination))/(np.sin(Solar_Zenith)*np.cos(Latitude)))  # radians
#    else:
#        Solar_Azimuth = -np.arccos((np.cos(Solar_Zenith)*np.sin(Latitude)-np.sin(Declination))/(np.sin(Solar_Zenith)*np.cos(Latitude))) # radians
    AOI = np.arccos(np.sin(Latitude)*np.sin(Declination)*np.cos(Tilt)+np.cos(Latitude)*np.sin(Declination)*np.sin(Tilt)*np.cos(Azimuth)+np.cos(Latitude)*np.cos(Declination)*np.cos(Tilt)*np.cos(Hour_Angle)-np.sin(Latitude)*np.cos(Declination)*np.sin(Tilt)*np.cos(Azimuth)*np.cos(Hour_Angle)-np.cos(Declination)*np.sin(Tilt)*np.sin(Hour_Angle)*np.sin(Azimuth))     # radians (Angle of Incidence)

    # Calculate radiation
    Incident_Beam = DNI*np.cos(AOI)
    Rel_Optical_Air_Mass = 1/(np.cos(Solar_Zenith)+0.50572*(96.07995-Solar_Zenith)**(-1.6364))
    Delta = DHI*Rel_Optical_Air_Mass/Extraterrestrial_Irradiance
    
    DHI_flag = (DHI==0)
    DHI[DHI_flag] += 1 # To avoid having 0's in DHI and 'RuntimeWarning: divide by zero encountered in true_divide'
    
    Epsilon = DHI_flag*(Kappa*Solar_Zenith**3)/(1+Kappa*Solar_Zenith**3) + (1-DHI_flag)*(np.nan_to_num((DHI+DNI)/DHI)+Kappa*Solar_Zenith**3)/(1+Kappa*Solar_Zenith**3)
# =============================================================================
#     if DHI == 0:
#         Epsilon = (Kappa*Solar_Zenith**3)/(1+Kappa*Solar_Zenith**3)
#     else:
#         Epsilon = ((DHI+DNI)/DHI+Kappa*Solar_Zenith**3)/(1+Kappa*Solar_Zenith**3)
# =============================================================================
    size_1 = Epsilon.size
    Epsilon_Period_1 = (Epsilon < 1.065).reshape(size_1,1)
    Epsilon_Period_2 = (Epsilon < 1.230).reshape(size_1,1)
    Epsilon_Period_3 = (Epsilon < 1.5).reshape(size_1,1)
    Epsilon_Period_4 = (Epsilon < 1.95).reshape(size_1,1)
    Epsilon_Period_5 = (Epsilon < 2.8).reshape(size_1,1)
    Epsilon_Period_6 = (Epsilon < 4.5).reshape(size_1,1)
    Epsilon_Period_7 = (Epsilon < 6.2).reshape(size_1,1)
    
    Perez_Coefficients = (Epsilon_Period_1* Perez_Bins[0] +
                         (1-Epsilon_Period_1)*Epsilon_Period_2 * Perez_Bins[1] +
                         (1-Epsilon_Period_2)*Epsilon_Period_3 * Perez_Bins[2] +
                         (1-Epsilon_Period_3)*Epsilon_Period_4 * Perez_Bins[3] +
                         (1-Epsilon_Period_4)*Epsilon_Period_5 * Perez_Bins[4] +
                         (1-Epsilon_Period_5)*Epsilon_Period_6 * Perez_Bins[5] +
                         (1-Epsilon_Period_6)*Epsilon_Period_7 * Perez_Bins[6] +
                         (1-Epsilon_Period_7) * Perez_Bins[7])
#    if Epsilon < 1.065:
#        Perez_Coefficients = Perez_Bins[0]
#    elif Epsilon < 1.230:
#        Perez_Coefficients = Perez_Bins[1]
#    elif Epsilon < 1.5:
#        Perez_Coefficients = Perez_Bins[2]
#    elif Epsilon < 1.95:
#        Perez_Coefficients = Perez_Bins[3]
#    elif Epsilon < 2.8:
#        Perez_Coefficients = Perez_Bins[4]
#    elif Epsilon < 4.5:
#        Perez_Coefficients = Perez_Bins[5]
#    elif Epsilon < 6.2:
#        Perez_Coefficients = Perez_Bins[6]
#    else:
#        Perez_Coefficients = Perez_Bins[7]

    F1_init = Perez_Coefficients[:,0]+Perez_Coefficients[:,1]*Delta+Perez_Coefficients[:,2]*Solar_Zenith
    F1 = (F1_init>=0)*F1_init
#    F1 = max(0, Perez_Coefficients[0]+Perez_Coefficients[1]*Delta+Perez_Coefficients[2]*Solar_Zenith)
    F2 = Perez_Coefficients[:,3]+Perez_Coefficients[:,4]*Delta+Perez_Coefficients[:,5]*Solar_Zenith
#    F2 = Perez_Coefficients[3]+Perez_Coefficients[4]*Delta+Perez_Coefficients[5]*Solar_Zenith

    a = np.cos(AOI) * (np.cos(AOI) >= 0)                                                                # radians
#    a = max(0, np.cos(AOI))                                                                            # radians
    b_1 = np.cos(85*np.pi/180)
    b_2 = np.cos(Solar_Zenith)
    b_mask = (b_1 > b_2)
    b = b_mask * b_1 + (1-b_mask) * b_2
#    b = max(np.cos(85*np.pi/180), np.cos(Solar_Zenith))                                                 # radians
    Incident_Diffuse = DHI*((1-F1)*(1+np.cos(Tilt))/2+F1*a/b+F2*np.sin(Tilt))                           # W/m2
    Incident_Ground = GHI*Albedo*(1-np.cos(Tilt))/2                                                     # W/m2
    Incident_POA = Incident_Beam + Incident_Diffuse + Incident_Ground                                   # W/m2

    # Calculate Module Cover Losses
    AOR = np.arcsin(1.0/n_Glass*np.sin(AOI))                                                                            # radians, Angle of Refraction from air to AR
    Transmittance_Glass = 1-0.5*np.sin(AOR-AOI)**2/np.sin(AOR+AOI)**2+np.tan(AOR-AOI)**2/np.tan(AOR+AOI)**2             # fraction, by Fresnel's law
    Transmittance_Cover = Transmittance_Glass                                                                           # fraction

    # Thermal Losses
    T_cell = Ambient_Temperature+Adsorption_Coefficient*Incident_POA*(1-Efficiency)/(U_0+U_1*Wind_Speed)                # deg C

    # Module Model
    Pdc = Incident_POA*Transmittance_Cover/1000*Pdc_0*(1+Gamma/100*(T_cell-T_ref))                        # W

    # System Losses
    Total_Loss = 1-(1-Soiling_Loss)*(1-Shading_Loss)*(1-Snow_Loss)*(1-Mismatch_Loss)*(1-Wiring_Loss)*(1-Connections_Loss)*(1-Light_Induced_Degradation)*(1-Nameplate_Rating)*(1-Age)*(1-Availability)
    Pdc = Pdc*(1-Total_Loss)                                                                            # W

    # Inverter Model
    Inverter_Pdc_0 = Inverter_Pac_0/Inverter_Eta_Nom
    Pdc_flag_1 = (Pdc == 0)
    
    Pdc_flag_2 = (Pdc < Inverter_Pdc_0)
    Pdc_flag_3 = (Pdc >= Inverter_Pdc_0)
    
    Xi = Pdc_flag_1*0 + (1-Pdc_flag_1)*(Pdc_flag_2)* Pdc/Inverter_Pdc_0 + (1-Pdc_flag_1)*(Pdc_flag_3) * 1 # Factor to calculate inverter efficiency
    Xi_mod = Xi.copy()
    Xi_mod[~(~Pdc_flag_1 & Pdc_flag_2)] += 1 # These indices of Xi will be discarded at the end. This way we avoid having zeros in these indices and getting runtime warning in the true division when calculating Inverter_Eta
    
    
    Inverter_Eta = Inverter_Eta_Nom/Inverter_Eta_Ref - 0.0162*Xi_mod - np.nan_to_num(0.0059/Xi_mod)   # Inverter Efficiency, Fraction
    Pac = Pdc_flag_1*0 + (1-Pdc_flag_1)*(Pdc_flag_2)* Inverter_Eta*Pdc + (1-Pdc_flag_1)*(Pdc_flag_3) * Inverter_Pac_0 # W, AC
    
# =============================================================================
#     if Pdc == 0:
#         Xi = 0 
# #        Inverter_Eta = 0
#         Pac = 0
#     elif Pdc < Inverter_Pdc_0:
#         Xi = Pdc/Inverter_Pdc_0                                                                         # Factor to calculate inverter efficiency
#         Inverter_Eta = Inverter_Eta_Nom/Inverter_Eta_Ref-0.0162*Xi-0.0059/Xi                            # Inverter Efficiency, Fraction
#         Pac = Inverter_Eta*Pdc                                                                          # W, AC
#     elif Pdc >= Inverter_Pdc_0:
#         Pac = Inverter_Pac_0                                                                            # W, AC
#         Xi = 1
# #    else: ## There shouldn't be any other cases!
# #        Pac = 0                                                                                         # W, AC
# #        Xi = 0
# =============================================================================

    # Calculate Total Cost and Maximum Production
    Num_Engines = PV_Roof_Area/Panel_Size                                                               # Modules
    Fuel_Input = (DNI+DHI+GHI)*PV_Roof_Area/1000     ## WHAT IS THIS?? The available solar irradiation  # kWh
    Hourly_Electrical_Consumption = 0
    Maximum_Heat_Production = 0  ## CAPACITY FOR SOLAR HEATING
    Maximum_Electrical_Production = Pac*Num_Engines/1000                                                # kWh
    Capital_Cost = Solar_Cost_SFH*Num_Engines*Pdc_0                                                     # USD
    Variable_Cost = 0
    Carbon_Emissions = 0
    Part_Load = Xi

    return [Fuel_Input, Hourly_Electrical_Consumption, Maximum_Heat_Production, Maximum_Electrical_Production, Capital_Cost, Variable_Cost, Carbon_Emissions, Num_Engines, Part_Load]
####### AUXILIARY FUNCTION END ##########
    
    

def Residential_Solar_1(Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude):

    ''' Based on NREL PV Watts v5 for a Standard PV module. http://www.nrel.gov/docs/fy14osti/62641.pdf. No tracking.
        Angles expected in degrees. System size expcted in kW. All other values from TMY file. Sizing of panel and inverter
        based on Enphase microinverter and single panel.
    '''
    Efficiency = 0.15                   # Percent
    Panel_Size = 1.733                  # m^2
#    Temp_Coefficient = -0.47            # %/deg-C
    Perez_Bins = [[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022],[0.13, 0.683, -0.151, -0.019, 0.066, -0.029], [0.33, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014], [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056], [1.06, -1.6, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]]
    Extraterrestrial_Irradiance = 1353  # W/m^2
    Kappa = 1.041                       # For Zenith in Radians
#    n_AR = 1.3                          # dimensionless
    n_Glass = 1.5                       # dimensionless
    T_ref = 25                          # deg C
    Pdc_0 = 260.0                       # Reference Power, W
    Gamma = -0.35                       # Percent/deg C (value for premium model from: http://www.nrel.gov/docs/fy14osti/62641.pdf)
    Adsorption_Coefficient = 0.9        # Fraction
    Soiling_Loss = 0.02                 # Fraction
    Shading_Loss = 0.03                 # Fraction
    Snow_Loss = 0.00                    # Fraction
    Mismatch_Loss = 0.02                # Fraction
    Wiring_Loss = 0.02                  # Fraction
    Connections_Loss = 0.005            # Fraction
    Light_Induced_Degradation = 0.015   # Fraction
    Nameplate_Rating = 0.01             # Fraction
    Age = 0.00                          # Fraction
    Availability = 0.03                 # Fraction
    Inverter_Eta_Ref = 0.9637           # Fraction
    Inverter_Eta_Nom = 0.96             # Fraction
    Inverter_Pac_0 = 240.0              # W, AC
    U_0 = 29                            # W/m^2-K
    U_1 = 0                             # W/m^2-K
    Solar_Cost_SFH = 2.71               # $/W (https://www.nrel.gov/solar/solar-installed-system-cost.html) # 2020

    return Computer(Efficiency, Panel_Size, Perez_Bins, Extraterrestrial_Irradiance, Kappa, n_Glass, T_ref, Pdc_0, Gamma, Adsorption_Coefficient, Soiling_Loss, Shading_Loss, Snow_Loss, Mismatch_Loss, Wiring_Loss, Connections_Loss, Light_Induced_Degradation, Nameplate_Rating, Age, Availability, Inverter_Eta_Ref, Inverter_Eta_Nom, Inverter_Pac_0, U_0, U_1, Solar_Cost_SFH, Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude)

def Residential_Solar_2(Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude):
    ''' Based on NREL PV Watts v5 for a Premium PV module. http://www.nrel.gov/docs/fy14osti/62641.pdf. No tracking.
        Angles expected in degrees. System size expcted in kW. All other values from TMY file. Sizing of panel and inverter
        based on Enphase microinverter and single panel.
    '''
    Efficiency = 0.19                   # Percent
    Panel_Size = 1.368                  # m^2
#    Temp_Coefficient = -0.35            # %/deg-C
    Perez_Bins = [[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022],[0.13, 0.683, -0.151, -0.019, 0.066, -0.029], [0.33, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014], [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056], [1.06, -1.6, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]]
    Extraterrestrial_Irradiance = 1353  # W/m^2
    Kappa = 1.041                       # For Zenith in Radians
#    n_AR = 1.3                          # dimensionless
    n_Glass = 1.5                       # dimensionless
    T_ref = 25                          # deg C
    Pdc_0 = 260.0                       # Reference Power, W
    Gamma = -0.35                       # Percent/deg C (value for premium model from: http://www.nrel.gov/docs/fy14osti/62641.pdf)
    Adsorption_Coefficient = 0.9        # Fraction
    Soiling_Loss = 0.02                 # Fraction
    Shading_Loss = 0.03                 # Fraction
    Snow_Loss = 0.00                    # Fraction
    Mismatch_Loss = 0.02                # Fraction
    Wiring_Loss = 0.02                  # Fraction
    Connections_Loss = 0.005            # Fraction
    Light_Induced_Degradation = 0.015   # Fraction
    Nameplate_Rating = 0.01             # Fraction
    Age = 0.00                          # Fraction
    Availability = 0.03                 # Fraction
    Inverter_Eta_Ref = 0.9637           # Fraction
    Inverter_Eta_Nom = 0.96             # Fraction
    Inverter_Pac_0 = 215.0              # W, AC
    U_0 = 29                            # W/m^2-K
    U_1 = 0                             # W/m^2-K
    Solar_Cost_SFH = 2.71               # $/W (https://www.nrel.gov/solar/solar-installed-system-cost.html) # 2020

    return Computer(Efficiency, Panel_Size, Perez_Bins, Extraterrestrial_Irradiance, Kappa, n_Glass, T_ref, Pdc_0, Gamma, Adsorption_Coefficient, Soiling_Loss, Shading_Loss, Snow_Loss, Mismatch_Loss, Wiring_Loss, Connections_Loss, Light_Induced_Degradation, Nameplate_Rating, Age, Availability, Inverter_Eta_Ref, Inverter_Eta_Nom, Inverter_Pac_0, U_0, U_1, Solar_Cost_SFH, Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude)

def Residential_Solar_3(Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude):
    ''' Based on NREL PV Watts v5 for a Thin-Film PV module. http://www.nrel.gov/docs/fy14osti/62641.pdf. No tracking.
        Angles expected in degrees. System size expcted in kW. All other values from TMY file. Sizing of panel and inverter
        based on Enphase microinverter and single panel.
    '''
    Efficiency = 0.10                   # Percent
    Panel_Size = 2.600                  # m^2
#    Temp_Coefficient = -0.20            # %/deg-C
    Perez_Bins = [[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022],[0.13, 0.683, -0.151, -0.019, 0.066, -0.029], [0.33, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014], [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056], [1.06, -1.6, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]]
    Extraterrestrial_Irradiance = 1353  # W/m^2
    Kappa = 1.041                       # For Zenith in Radians
#    n_AR = 1.3                          # dimensionless
    n_Glass = 1.5                       # dimensionless
    T_ref = 25                          # deg C
    Pdc_0 = 260.0                       # Reference Power, W
    Gamma = -0.35                       # Percent/deg C (value for premium model from: http://www.nrel.gov/docs/fy14osti/62641.pdf)
    Adsorption_Coefficient = 0.9        # Fraction
    Soiling_Loss = 0.02                 # Fraction
    Shading_Loss = 0.03                 # Fraction
    Snow_Loss = 0.00                    # Fraction
    Mismatch_Loss = 0.02                # Fraction
    Wiring_Loss = 0.02                  # Fraction
    Connections_Loss = 0.005            # Fraction
    Light_Induced_Degradation = 0.015   # Fraction
    Nameplate_Rating = 0.01             # Fraction
    Age = 0.00                          # Fraction
    Availability = 0.03                 # Fraction
    Inverter_Eta_Ref = 0.9637           # Fraction
    Inverter_Eta_Nom = 0.96             # Fraction
    Inverter_Pac_0 = 215.0              # W, AC
    U_0 = 29                            # W/m^2-K
    U_1 = 0                             # W/m^2-K
    Solar_Cost_SFH = 2.71               # $/W (https://www.nrel.gov/solar/solar-installed-system-cost.html) # 2020

    return Computer(Efficiency, Panel_Size, Perez_Bins, Extraterrestrial_Irradiance, Kappa, n_Glass, T_ref, Pdc_0, Gamma, Adsorption_Coefficient, Soiling_Loss, Shading_Loss, Snow_Loss, Mismatch_Loss, Wiring_Loss, Connections_Loss, Light_Induced_Degradation, Nameplate_Rating, Age, Availability, Inverter_Eta_Ref, Inverter_Eta_Nom, Inverter_Pac_0, U_0, U_1, Solar_Cost_SFH, Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude)

def Commercial_Solar_1(Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude):
    ''' Based on NREL PV Watts v5 for a Standard PV module. http://www.nrel.gov/docs/fy14osti/62641.pdf. No tracking.
        Angles expected in degrees. System size expcted in kW. All other values from TMY file. Sizing of panel and inverter
        based on Enphase microinverter and single panel.
    '''
    Efficiency = 0.15                   # Percent
    Panel_Size = 1.733                  # m^2
#    Temp_Coefficient = -0.47            # %/deg-C
    Perez_Bins = [[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022],[0.13, 0.683, -0.151, -0.019, 0.066, -0.029], [0.33, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014], [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056], [1.06, -1.6, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]]
    Extraterrestrial_Irradiance = 1353  # W/m^2
    Kappa = 1.041                       # For Zenith in Radians
#    n_AR = 1.3                          # dimensionless
    n_Glass = 1.5                       # dimensionless
    T_ref = 25                          # deg C
    Pdc_0 = 260.0                       # Reference Power, W
    Gamma = -0.35                       # Percent/deg C (value for premium model from: http://www.nrel.gov/docs/fy14osti/62641.pdf)
    Adsorption_Coefficient = 0.9        # Fraction
    Soiling_Loss = 0.02                 # Fraction
    Shading_Loss = 0.03                 # Fraction
    Snow_Loss = 0.00                    # Fraction
    Mismatch_Loss = 0.02                # Fraction
    Wiring_Loss = 0.02                  # Fraction
    Connections_Loss = 0.005            # Fraction
    Light_Induced_Degradation = 0.015   # Fraction
    Nameplate_Rating = 0.01             # Fraction
    Age = 0.00                          # Fraction
    Availability = 0.03                 # Fraction
    Inverter_Eta_Ref = 0.9637           # Fraction
    Inverter_Eta_Nom = 0.96             # Fraction
    Inverter_Pac_0 = 215.0              # W, AC
    Adsorption_Coefficient = 0.9        # dimensionless
    U_0 = 29                            # W/m^2-K
    U_1 = 0                             # W/m^2-K
    U_0 = 29                            # W/m^2-K
    U_1 = 0                             # W/m^2-K
    Solar_Cost_SFH = 1.72               # $/W (https://www.nrel.gov/solar/solar-installed-system-cost.html) #2020

    return Computer(Efficiency, Panel_Size, Perez_Bins, Extraterrestrial_Irradiance, Kappa, n_Glass, T_ref, Pdc_0, Gamma, Adsorption_Coefficient, Soiling_Loss, Shading_Loss, Snow_Loss, Mismatch_Loss, Wiring_Loss, Connections_Loss, Light_Induced_Degradation, Nameplate_Rating, Age, Availability, Inverter_Eta_Ref, Inverter_Eta_Nom, Inverter_Pac_0, U_0, U_1, Solar_Cost_SFH, Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude)

def Commercial_Solar_2(Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude):
    ''' Based on NREL PV Watts v5 for a Premium PV module. http://www.nrel.gov/docs/fy14osti/62641.pdf. No tracking.
        Angles expected in degrees. System size expcted in kW. All other values from TMY file. Sizing of panel and inverter
        based on Enphase microinverter and single panel.
    '''
    Efficiency = 0.19                   # Percent
    Panel_Size = 1.368                  # m^2
#    Temp_Coefficient = -0.35            # %/deg-C
    Perez_Bins = [[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022],[0.13, 0.683, -0.151, -0.019, 0.066, -0.029], [0.33, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014], [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056], [1.06, -1.6, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]]
    Extraterrestrial_Irradiance = 1353  # W/m^2
    Kappa = 1.041                       # For Zenith in Radians
#    n_AR = 1.3                          # dimensionless
    n_Glass = 1.5                       # dimensionless
    T_ref = 25                          # deg C
    Pdc_0 = 260.0                       # Reference Power, W
    Gamma = -0.35                       # Percent/deg C (value for premium model from: http://www.nrel.gov/docs/fy14osti/62641.pdf)
    Adsorption_Coefficient = 0.9        # Fraction
    Soiling_Loss = 0.02                 # Fraction
    Shading_Loss = 0.03                 # Fraction
    Snow_Loss = 0.00                    # Fraction
    Mismatch_Loss = 0.02                # Fraction
    Wiring_Loss = 0.02                  # Fraction
    Connections_Loss = 0.005            # Fraction
    Light_Induced_Degradation = 0.015   # Fraction
    Nameplate_Rating = 0.01             # Fraction
    Age = 0.00                          # Fraction
    Availability = 0.03                 # Fraction
    Inverter_Eta_Ref = 0.9637           # Fraction
    Inverter_Eta_Nom = 0.96             # Fraction
    Inverter_Pac_0 = 215.0              # W, AC
    U_0 = 29                            # W/m^2-K
    U_1 = 0                             # W/m^2-K
    Solar_Cost_SFH = 1.72               # $/W (https://www.nrel.gov/solar/solar-installed-system-cost.html) #2020

    return Computer(Efficiency, Panel_Size, Perez_Bins, Extraterrestrial_Irradiance, Kappa, n_Glass, T_ref, Pdc_0, Gamma, Adsorption_Coefficient, Soiling_Loss, Shading_Loss, Snow_Loss, Mismatch_Loss, Wiring_Loss, Connections_Loss, Light_Induced_Degradation, Nameplate_Rating, Age, Availability, Inverter_Eta_Ref, Inverter_Eta_Nom, Inverter_Pac_0, U_0, U_1, Solar_Cost_SFH, Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude)

def Commercial_Solar_3(Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude):
    ''' Based on NREL PV Watts v5 for a Thin-Film PV module. http://www.nrel.gov/docs/fy14osti/62641.pd.f No tracking.
        Angles expected in degrees. System size expcted in kW. All other values from TMY file. Sizing of panel and inverter
        based on Enphase microinverter and single panel.
    '''
    Efficiency = 0.10                   # Percent
    Panel_Size = 2.600                  # m^2
#    Temp_Coefficient = -0.20            # %/deg-C
    Perez_Bins = [[-0.008, 0.588, -0.062, -0.06, 0.072, -0.022],[0.13, 0.683, -0.151, -0.019, 0.066, -0.029], [0.33, 0.487, -0.221, 0.055, -0.064, -0.026], [0.568, 0.187, -0.295, 0.109, -0.152, -0.014], [0.873, -0.392, -0.362, 0.226, -0.462, 0.001], [1.132, -1.237, -0.412, 0.288, -0.823, 0.056], [1.06, -1.6, -0.359, 0.264, -1.127, 0.131], [0.678, -0.327, -0.25, 0.156, -1.377, 0.251]]
    Extraterrestrial_Irradiance = 1353  # W/m^2
    Kappa = 1.041                       # For Zenith in Radians
#    n_AR = 1.3                          # dimensionless
    n_Glass = 1.5                       # dimensionless
    T_ref = 25                          # deg C
    Pdc_0 = 260.0                       # Reference Power, W
    Gamma = -0.35                       # Percent/deg C (value for premium model from: http://www.nrel.gov/docs/fy14osti/62641.pdf)
    Adsorption_Coefficient = 0.9        # Fraction
    Soiling_Loss = 0.02                 # Fraction
    Shading_Loss = 0.03                 # Fraction
    Snow_Loss = 0.00                    # Fraction
    Mismatch_Loss = 0.02                # Fraction
    Wiring_Loss = 0.02                  # Fraction
    Connections_Loss = 0.005            # Fraction
    Light_Induced_Degradation = 0.015   # Fraction
    Nameplate_Rating = 0.01             # Fraction
    Age = 0.00                          # Fraction
    Availability = 0.03                 # Fraction
    Inverter_Eta_Ref = 0.9637           # Fraction
    Inverter_Eta_Nom = 0.96             # Fraction
    Inverter_Pac_0 = 215.0              # W, AC
    U_0 = 29                            # W/m^2-K
    U_1 = 0                             # W/m^2-K
    Solar_Cost_SFH = 1.72               # $/W (https://www.nrel.gov/solar/solar-installed-system-cost.html) #2020

    return Computer(Efficiency, Panel_Size, Perez_Bins, Extraterrestrial_Irradiance, Kappa, n_Glass, T_ref, Pdc_0, Gamma, Adsorption_Coefficient, Soiling_Loss, Shading_Loss, Snow_Loss, Mismatch_Loss, Wiring_Loss, Connections_Loss, Light_Induced_Degradation, Nameplate_Rating, Age, Availability, Inverter_Eta_Ref, Inverter_Eta_Nom, Inverter_Pac_0, U_0, U_1, Solar_Cost_SFH, Hour, UTC, PV_Roof_Area, Tilt, Azimuth, Latitude, Longitude, DNI, DHI, GHI, Albedo, Ambient_Temperature, Wind_Speed, Altitude)