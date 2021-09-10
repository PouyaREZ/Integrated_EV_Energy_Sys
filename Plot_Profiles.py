# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:30:10 2021

@author: PouyaRZ
"""

from Main import SupplyandDemandOptimization



neighborhood1_var_inputs = [39,0,2,42,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,0,32,5,15,0]
'''
Building 1	Building 2	Building 3	Building 4	Building 5	Building 6	Building 7	Building 8	Building 9	Building 10	Building 11	Building 12	Building 13	Building 14	Building 15	Building 16	Building 17	Building 18	Building 19	Building 20	Building 21	Engine_Var	Chiller_Var	Comm_Solar_Var	EV_Load_Ratio	Total_Electricity_Demand	LCC_Total	Total_GHG	LCC (w/o purchase from grid for EV)	Res_Area_Percents 1	Office_Area_Percents 2	Comm_Area_Percents 3	Warehouse_Area_Percents 4	Hospital_Area_Percents 5	Hotel_Area_Percents 6	Educ_Area_Percents 7	Site_FAR	Average_Height	Total_GFA	GHG (w/o grid for EV)	Total_Electric_Load_on_CCHP	Peak_Electric_Load_on_CCHP	Peak_Electricity_Demand	Total_EV_Demand	Peak_EV_Demand	Overall_Efficiency	Total E Demand bldgs / Total EV Demand	Peak E Demand bldgs / Peak EV Demand
39	0	2	42	52	0	0	0	0	0	0	0	0	0	0	0	0	0	0	17	0	32	5	15	0	55539158.96	255825006.8	60029.74593	254900198.1	0.177733	0.817744	0.004523	0	0	0	0	10.503004	105.947368	2945470	43262.72549	73307150.37	27735.40068	10932.33277	2163870.832	609.194212	0.154609	26	18
'''
SupplyandDemandOptimization(neighborhood1_var_inputs, plot_profiles=True)


neighborhood2 = [379,468,178,268,3,490,187,191,115,987,175,116,64,346,29,16,640,12,94,337,1,25,16,28,0]
'''
Building 1	Building 2	Building 3	Building 4	Building 5	Building 6	Building 7	Building 8	Building 9	Building 10	Building 11	Building 12	Building 13	Building 14	Building 15	Building 16	Building 17	Building 18	Building 19	Building 20	Building 21	Engine_Var	Chiller_Var	Comm_Solar_Var	EV_Load_Ratio	Total_Electricity_Demand	LCC_Total	Total_GHG	LCC (w/o purchase from grid for EV)	Res_Area_Percents 1	Office_Area_Percents 2	Comm_Area_Percents 3	Warehouse_Area_Percents 4	Hospital_Area_Percents 5	Hotel_Area_Percents 6	Educ_Area_Percents 7	Site_FAR	Average_Height	Total_GFA	GHG (w/o grid for EV)	Total_Electric_Load_on_CCHP	Peak_Electric_Load_on_CCHP	Peak_Electricity_Demand	Total_EV_Demand	Peak_EV_Demand	Overall_Efficiency	Total E Demand bldgs / Total EV Demand	Peak E Demand bldgs / Peak EV Demand
379	468	178	268	3	490	187	191	115	987	175	116	64	346	29	16	640	12	94	337	1	25	16	28	0	3016995147	10437080792	1231980.487	10413719804	0.388355	0.130862	0.094407	0.014898	0.194585	0.13448	0.042414	3.101567	44.347724	20771147	827307.5055	3205142643	577746.0823	546346.9511	56341962.97	18562.84242	0.187252	54	29
'''
SupplyandDemandOptimization(neighborhood1_var_inputs, plot_profiles=True)
