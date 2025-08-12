# %%  Import packages
from matplotlib import colors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime 
import os
filename = datetime.datetime.now()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%% Load water quality

Salinity = pd.read_csv(r"OUT\S.csv",header = None)
Salinity = pd.DataFrame(np.fliplr(Salinity))
Salinity = Salinity.iloc[:-1,2:103]
time_step = pd.date_range('2017-01-01', periods=Salinity.shape[0], freq='1H')
Salinity["Date"] = time_step
Salinity=Salinity.set_index("Date")
Salinity = Salinity.resample('D').mean()
Salinity = Salinity.melt(var_name='Cell',ignore_index = False)
Salinity = Salinity.astype(float)
Salinity["Location"] = Salinity["Cell"]*2-4
Salinity = Salinity.rename(columns={'value': 'Salinity'})
Salinity = Salinity.reset_index()
Salinity['Month'] = Salinity["Date"].dt.month
Salinity['Season'] = "Wet"
Salinity.loc[Salinity['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
Salinity["Type"] = "Simulation"

Surface = pd.read_csv(r"OUT\Hydrodynamics\surface.csv",header = None)
Surface = pd.DataFrame(np.fliplr(Surface))
Surface = Surface.iloc[:-1,1:102]
Surface["Date"] = time_step
Surface=Surface.set_index("Date")
Surface = Surface.resample('D').mean()
Surface = Surface.melt(var_name='Cell',ignore_index = False)
Surface = Surface.astype(float)
Surface["Location"] = Surface["Cell"]*2-2
Surface = Surface.rename(columns={'value': 'Surface'})
Surface = Surface.reset_index()
Surface['Month'] = Surface["Date"].dt.month
Surface['Season'] = "Wet"
Surface.loc[Surface['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
Surface["Type"] = "Simulation"

NO3 = pd.read_csv(r"OUT\NO3.csv",header = None)
NO3 = pd.DataFrame(np.fliplr(NO3))
NO3 = NO3.iloc[:-1,2:103]
NO3["Date"] = time_step
NO3=NO3.set_index("Date")
NO3 = NO3.resample('D').mean()
NO3 = NO3.melt(var_name='Cell',ignore_index = False)
NO3 = NO3.astype(float)
NO3["Location"] = NO3["Cell"]*2-4
NO3 = NO3.rename(columns={'value': 'NO3'})
NO3 = NO3.reset_index()
NO3['Month'] = NO3["Date"].dt.month
NO3['Season'] = "Wet"
NO3.loc[NO3['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
NO3["Type"] = "Simulation"


TOC = pd.read_csv(r"OUT\TOC.csv",header = None)
TOC = pd.DataFrame(np.fliplr(TOC))
TOC = TOC.iloc[:-1,2:103]
TOC["Date"] = time_step
TOC=TOC.set_index("Date")
TOC = TOC.resample('D').mean()
TOC = TOC.melt(var_name='Cell',ignore_index = False)
TOC = TOC.astype(float)
TOC["Location"] = TOC["Cell"]*2-4
TOC = TOC.rename(columns={'value': 'TOC'})
TOC = TOC.reset_index()
TOC['Month'] = TOC["Date"].dt.month
TOC['Season'] = "Wet"
TOC.loc[TOC['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
TOC["Type"] = "Simulation"
 
DO = pd.read_csv(r"OUT\O2.csv",header = None)
DO = pd.DataFrame(np.fliplr(DO))
DO = DO.iloc[:-1,2:103]
DO["Date"] = time_step
DO=DO.set_index("Date")
DO = DO.resample('D').mean()
DO = DO.melt(var_name='Cell',ignore_index = False)
DO = DO.astype(float)
DO["Location"] = DO["Cell"]*2-4
DO = DO.rename(columns={'value': 'DO'})
DO = DO.reset_index()
DO['Month'] = DO["Date"].dt.month
DO['Season'] = "Wet"
DO.loc[DO['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
DO["Type"] = "Simulation"

PO4 = pd.read_csv(r"OUT\PO4.csv",header = None)
PO4 = pd.DataFrame(np.fliplr(PO4))
PO4 = PO4.iloc[:-1,2:103]
PO4["Date"] = time_step
PO4=PO4.set_index("Date")
PO4 = PO4.resample('D').mean()
PO4 = PO4.melt(var_name='Cell',ignore_index = False)
PO4 = PO4.astype(float)
PO4["Location"] = PO4["Cell"]*2-4
PO4 = PO4.rename(columns={'value': 'PO4'})
PO4 = PO4.reset_index()
PO4['Month'] = PO4["Date"].dt.month
PO4['Season'] = "Wet"
PO4.loc[PO4['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
PO4["Type"] = "Simulation"

NH4 = pd.read_csv(r"OUT\NH4.csv",header = None)
NH4 = pd.DataFrame(np.fliplr(NH4))
NH4 = NH4.iloc[:-1,2:103]
NH4["Date"] = time_step
NH4 = NH4.set_index("Date")
NH4 = NH4.resample('D').mean()
NH4 = NH4.melt(var_name='Cell',ignore_index = False)
NH4 = NH4.astype(float)
NH4["Location"] = NH4["Cell"]*2-4
NH4 = NH4.rename(columns={'value': 'NH4'})
NH4 = NH4.reset_index()
NH4['Month'] = NH4["Date"].dt.month
NH4['Season'] = "Wet"
NH4.loc[NH4['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
NH4["Type"] = "Simulation"

DIA = pd.read_csv(r"OUT\Dia.csv",header = None)
DIA = pd.DataFrame(np.fliplr(DIA))
DIA = DIA.iloc[:-1,2:103]
DIA["Date"] = time_step
DIA=DIA.set_index("Date")
DIA = DIA.resample('D').mean()
DIA = DIA.melt(var_name='Cell',ignore_index = False)
DIA = DIA.astype(float)
DIA["Location"] = DIA["Cell"]*2-4
DIA = DIA.rename(columns={'value': 'Diatom'})
DIA = DIA.reset_index()
DIA['Month'] = DIA["Date"].dt.month
DIA['Season'] = "Wet"
DIA.loc[DIA['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
DIA["Type"] = "Simulation"

TSS = pd.read_csv(r"OUT\SPM.csv",header = None)
TSS = pd.DataFrame(np.fliplr(TSS))
TSS = TSS.iloc[:-1,2:103]
TSS["Date"] = time_step
TSS=TSS.set_index("Date")
TSS = TSS.resample('D').mean()
TSS = TSS.melt(var_name='Cell',ignore_index = False)
TSS = TSS.astype(float)
TSS["Location"] = TSS["Cell"]*2-4
TSS = TSS.rename(columns={'value': 'TSS'})
TSS = TSS.reset_index()
TSS['Month'] = TSS["Date"].dt.month
TSS['Season'] = "Wet"
TSS.loc[TSS['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
TSS["Type"] = "Simulation"

DSi = pd.read_csv(r"OUT\Si.csv",header = None)
DSi = pd.DataFrame(np.fliplr(DSi))
DSi = DSi.iloc[:-1,2:103]
DSi["Date"] = time_step
DSi=DSi.set_index("Date")
DSi = DSi.resample('D').mean()
DSi = DSi.melt(var_name='Cell',ignore_index = False)
DSi = DSi.astype(float)
DSi["Location"] = DSi["Cell"]*2-4
DSi = DSi.rename(columns={'value': 'DSi'})
DSi = DSi.reset_index()
DSi['Month'] = DSi["Date"].dt.month
DSi['Season'] = "Wet"
DSi.loc[DSi['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
DSi["Type"] = "Simulation"

# select parameters

model_data = pd.concat([Surface,NH4,NO3,PO4,DSi,DIA,TOC,DO,TSS,Salinity],axis = 1,ignore_index=False, sort=False)

simulation = model_data[["Location","Date","Surface","NH4","NO3","PO4","DSi","Diatom","TOC","DO","TSS","Salinity"]]
simulation = simulation.loc[:, ~simulation.columns[::-1].duplicated()[::-1]] # remove duplicate columns
#simulation_mean = simulation.groupby("Location").mean().reset_index()

#%% Load observation

# CEM DATA
obs_CEM = pd.read_excel(r"OUT\PlotINPUT\CEM_2017-2018.xlsx",engine='openpyxl',sheet_name="Data")

# Calculate daily values (average of high and low tide)
obs_CEM = obs_CEM.groupby(["Date","Location"]).mean().reset_index()
obs_CEM[["Date"]] = obs_CEM[["Date"]].apply(pd.to_datetime)

# CARE DATA
obs_CARE = pd.read_excel(r"OUT\PlotINPUT\CARE_2017-2018.xlsx",engine='openpyxl',sheet_name="Data")
obs_CARE[["Date"]] = obs_CARE[["Date"]].apply(pd.to_datetime)

#%% Plot longitudinal profile
parameter = ["TSS (mg/L)","DO (mg/L)","NH4 (mgN/L)","NO3 (mgN/L)","PO4 (mgP/L)","DSi (mgSi/L)","Chl-a (μg/L)","TOC (mgC/L)"]

n=len(parameter)
fig2 = plt.figure(figsize=(16,12)) # Notice the equal aspect ratio
ax = [fig2.add_subplot(np.intc(np.ceil(n/2)),2,i+1) for i in range(np.intc(2*np.ceil(n/2)))]

for i,para in enumerate(parameter):
    try:
        selected_parameter = parameter[i]
        sns.lineplot(x='Location',y=selected_parameter,data=simulation,ax=ax[i],ci="sd")
        
        sns.scatterplot(x="Location", y=selected_parameter,label='Observation',facecolor="none",edgecolor="red",
                    data=obs_CARE,legend = False ,ax=ax[i])

        ax[i].set_ylabel(selected_parameter)

        ax[i].plot(obs_CEM['Location'],obs_CEM[selected_parameter],'r.')
    except:
        pass

ax[0].set_ylim(0, 200)
ax[6].set_ylim(0, 76)
ax[0].legend(["Simulation","Observation"])
fig2.suptitle("Water quality in Saigon River 1/2017-1/2019", y=0.92,fontsize=15)
fig2.subplots_adjust(wspace=0.15, hspace=0)
fig2.savefig("OUT\\Longitudinal Profile %s.png" %filename.strftime("%d %m %H-%M"),bbox_inches='tight',dpi=200)


#%% Plot Each station =  8x3 figures
simulation_PC = simulation.loc[simulation.loc[:,"Location"]==86,:]
simulation_BD = simulation.loc[simulation.loc[:,"Location"]==130,:]
simulation_BK = simulation.loc[simulation.loc[:,"Location"]==156,:]

observation_PC = obs_CARE[obs_CARE.Site=="PC"]
observation_BD = obs_CARE[obs_CARE.Site=="BD"]
observation_BK = obs_CARE[obs_CARE.Site=="BK"]

simulation_PC.loc[:,'Month'] = simulation_PC["Date"].dt.month
simulation_BD.loc[:,'Month'] = simulation_BD["Date"].dt.month
simulation_BK.loc[:,'Month'] = simulation_PC["Date"].dt.month

simulation_PC_mean = simulation_PC.groupby(by="Month").mean()
simulation_BD_mean = simulation_BD.groupby(by="Month").mean()
simulation_BK_mean = simulation_BK.groupby(by="Month").mean()

parameter = ["TSS (mg/L)","DO (mg/L)","NH4 (mgN/L)","NO3 (mgN/L)","PO4 (mgP/L)","DSi (mgSi/L)","Chl-a (μg/L)","TOC (mgC/L)"]
n=len(parameter)

fig = plt.figure(figsize=(12,12)) # Notice the equal aspect ratio
ax = [fig.add_subplot(8,3,i+1) for i in range(24)]

for i,para in enumerate(parameter):
    try: #using try because missing some parameters in CEM
        selected_parameter = parameter[i]
        simulation_PC.plot(x="Date", y=selected_parameter,ax=ax[3*i],label="Simulated PC",lw=0.5,color="black", legend=None)
        simulation_BD.plot(x="Date", y=selected_parameter,ax=ax[3*i+1],label="Simulated BD",lw=0.5,color="black",legend=None)
        simulation_BK.plot(x="Date", y=selected_parameter,ax=ax[3*i+2],label="Simulated BK",lw=0.5,color="black",legend=None)

        max_lim = np.max([simulation_PC[selected_parameter],simulation_BD[selected_parameter],simulation_BK[selected_parameter]])
        min_lim = np.min([simulation_PC[selected_parameter],simulation_BD[selected_parameter],simulation_BK[selected_parameter]])

        ax[3*i].set_ylim(0, max_lim)
        ax[3*i+1].set_ylim(0, max_lim)
        ax[3*i+2].set_ylim(0, max_lim)

        sns.scatterplot(x="Date", y=selected_parameter,label='Observed PC',facecolor="none",edgecolor="blue",marker='.',
                    data=observation_PC,legend = False ,ax=ax[3*i],zorder=10)
        
        sns.scatterplot(x="Date", y=selected_parameter,label='Observed BD',facecolor="none",edgecolor="blue",marker='.',zorder=10,
                    data=observation_BD,legend = False ,ax=ax[3*i+1])

        sns.scatterplot(x="Date", y=selected_parameter,label='Observed BK',facecolor="none",edgecolor="blue",marker='.',zorder=10,
                    data=observation_BK,legend = False ,ax=ax[3*i+2])
    except:
        pass

for i in range(8):
    ax[3*i].set_ylabel((parameter[i]))
    ax[3*i+1].set_ylabel("")
    ax[3*i+2].set_ylabel("")

    ax[3*i+1].set_yticks([])
    ax[3*i+2].set_yticks([])

    ax[3*i].set_xticks([])
    ax[3*i+1].set_xticks([])
    ax[3*i+2].set_xticks([])

    ax[3*i].set_xlabel("")
    ax[3*i+1].set_xlabel("")
    ax[3*i+2].set_xlabel("")


ax[21].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[21].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax[21].tick_params(axis='x', rotation=90)

ax[22].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[22].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax[22].tick_params(axis='x', rotation=90)

ax[23].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[23].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax[23].tick_params(axis='x', rotation=90)

ax[0].title.set_text("km 86 - upstream station")
ax[1].title.set_text("km 130 - urban station")
ax[2].title.set_text("km 156 - downstream station")

fig.subplots_adjust(wspace=0.05, hspace=0)
fig.savefig("OUT\\Water quality %s.png" % (filename.strftime("%d-%m-%H-%M")),bbox_inches='tight',dpi=200)
# %%
