# %%  Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime 
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr
filename = datetime.datetime.now()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%% Load Geometry
# Water Level C-GEM

water_level = pd.read_csv(r"OUT\Hydrodynamics\PROF.csv",header = None)
water_level = pd.DataFrame(np.fliplr(water_level))
water_level = water_level.iloc[:,2:103].rename_axis(index="hour")
water_level = water_level.melt(var_name='Cell')
water_level = water_level.astype(float)
water_level["Location"] = water_level["Cell"]*2-4
water_level = water_level.rename(columns={'value': 'Level'})

# Depth C-GEM
river_bed = pd.read_csv(r"OUT\Hydrodynamics\slope.csv",header = None)
river_bed = pd.DataFrame(np.fliplr(river_bed))
river_bed = river_bed.iloc[:,3:104].rename_axis(index="hour")
river_bed = river_bed.melt(var_name='Cell')
river_bed = river_bed.astype(float)
river_bed["Location"] = river_bed["Cell"]*2-6
river_bed = river_bed.rename(columns={'value': 'Depth'})
river_bed["Depth"] = -river_bed["Depth"]

# Width C-GEM
river_width = pd.read_csv(r"OUT\Hydrodynamics\B.csv",header = None)
river_width = pd.DataFrame(np.fliplr(river_width))
river_width = river_width.iloc[:,3:104].rename_axis(index="hour")
river_width = river_width.melt(var_name='Cell')
river_width = river_width.astype(float)
river_width["Location"] = river_width["Cell"]*2-6
river_width = river_width.rename(columns={'value': 'Width'})

# Read observation data river depth and width
data_river = pd.read_csv(r"SIWRR_river_depth_width.csv",index_col=None) 
data_river = data_river.apply(pd.to_numeric, errors='ignore')
data_river_mean=data_river.groupby('Location').mean()
data_river_mean.reset_index(level=0, inplace=True)

#%% Tidal range observation
obs_tidal_range = pd.read_excel(r"WACC-Tidal-range2017-2018.xlsx",engine='openpyxl')
obs_tidal_range[["Day"]] = obs_tidal_range[["Day"]].apply(pd.to_datetime)

obs_tidal_range_month = pd.read_excel(r"SIHYMECC_Tidal-range2017-2018.xlsx",engine='openpyxl')
obs_tidal_range_month[["Day"]] = obs_tidal_range_month[["Day"]].apply(pd.to_datetime)

#%% Tidal range model
CEM_tidal_range = pd.read_csv(r"CEM-Tidal-range.csv",index_col=None)
CEM_tidal_range = CEM_tidal_range.apply(pd.to_numeric, errors='ignore')
CEM_tidal_range["Tidal Range"] = CEM_tidal_range["Tidal Range"]/100

# Read water level from model
tide_model = pd.read_csv(r"OUT\Hydrodynamics\PROF.csv",header = None)
tide_model = pd.DataFrame(np.fliplr(tide_model))
tide_model = tide_model.iloc[:-1,2:103].rename_axis(index="hour")
time_step = pd.date_range(obs_tidal_range.Day[0], periods=tide_model.shape[0], freq='1H')
tide_model["Date"]=time_step
tide_model['Day'] = tide_model["Date"].dt.date
model_tidal_range = tide_model.groupby('Day').max()-tide_model.groupby('Day').min()
model_tidal_range.reset_index(level=0, inplace=True)

# Create tidal profile 
model_tide_profile = model_tidal_range.drop(columns = model_tidal_range.columns[-1]).set_index("Day")
model_tide_profile = model_tide_profile.rename_axis("Day", axis="index").rename_axis("Cell", axis="columns")
model_tide_profile_long = model_tide_profile.melt(ignore_index = False)
model_tide_profile_long.reset_index(level=0, inplace=True)
model_tide_profile_long["Location"]=model_tide_profile_long["Cell"]*2-4

# Combine model and observation tidal range
compare_model_full = pd.concat([obs_tidal_range, model_tidal_range], axis=1, keys="Day").dropna()
compare_model_full = compare_model_full.T.groupby(level=1).first().T.drop(["Date"],axis=1)

compare_model_month = pd.concat([obs_tidal_range_month, model_tidal_range], axis=1, keys="Day").dropna()
compare_model_month = compare_model_month.T.groupby(level=1).first().T.drop(["Date"],axis=1)

#%% Load Salinity
Salinity = pd.read_csv(r"OUT\S.csv",header = None)
Salinity = pd.DataFrame(np.fliplr(Salinity))
Salinity = Salinity.iloc[:-1,2:103]
Salinity["Date"] = time_step
Salinity=Salinity.set_index("Date")
# Hourly to daily salinity
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

# Observation data from CEM 2013 - 2016
Salinity_CEM = pd.read_csv(r"CEM_quality_2014-2016.csv")
Salinity_CEM[["Date"]] = Salinity_CEM[["Date"]].apply(pd.to_datetime)
Salinity_CEM['Month'] = Salinity_CEM["Date"].dt.month
Salinity_CEM['Season'] = "Wet"
Salinity_CEM.loc[Salinity_CEM['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'
Salinity_CEM["Type"] = "Observation"

Salinity_combine = pd.concat([Salinity,Salinity_CEM],axis="index",keys = "Date")

# Read dispersion
disp = pd.read_csv(r"OUT\Hydrodynamics\disp.csv",header = None)
disp = pd.DataFrame(np.fliplr(disp))
disp = disp.iloc[:-1,2:103]
disp["Date"] = time_step
disp=disp.set_index("Date")
disp = disp.melt(var_name='Cell',ignore_index = False)
disp = disp.astype(float)
disp["Location"] = disp["Cell"]*2-4
disp = disp.rename(columns={'value': 'Dispersion'})
disp = disp.reset_index()
disp['Month'] = disp["Date"].dt.month
disp['Season'] = "Wet"
disp.loc[disp['Month'].isin([12,1,2,3,4,5]),'Season'] = 'Dry'

##############----------------------###############
#%% Plot Tidal Range and Geometry


#### Profile of Geometry and Tidal Range, Salinity
fig = plt.figure(figsize=(16,12)) # Notice the equal aspect ratio
ax = [fig.add_subplot(5,2,i+1) for i in range(10)]

sns.lineplot(x="Location", y="value",data = model_tide_profile_long, ci="sd",ax=ax[4],label="Simulation")

# CEM Salinity Plot dry season
sns.lineplot(x="Location",y="Salinity",data=Salinity_combine[Salinity_combine["Season"]=="Dry"],hue="Type",ci="sd",ax=ax[6])
# CEM Salinity Plot wet season
sns.lineplot(x="Location",y="Salinity",data=Salinity_combine[Salinity_combine["Season"]=="Wet"],hue="Type",ci="sd",ax=ax[8])

sns.lineplot(x="Location",y="Dispersion",data=disp,hue="Season",ci="sd",ax=ax[9])

model_tidal_range.plot(x="Day", y=43,ax=ax[1],label="PC_Simulation")
model_tidal_range.plot(x="Day", y=65,ax=ax[3],label="BD_Simulation")
model_tidal_range.plot(x="Day", y=77,ax=ax[5],label="BK_Simulation")

obs_tidal_range.plot(x="Day", y="PC",ax=ax[1],label="PC_Observation")
obs_tidal_range.plot(x="Day", y="BD",ax=ax[3],label="BD_Observation")
obs_tidal_range.plot(x="Day", y="BK",ax=ax[5],label="BK_Observation")

ax[1].plot(obs_tidal_range_month['Day'], obs_tidal_range_month['PC'],'r.',label = "PC_Observation_SIHYMECC")
ax[3].plot(obs_tidal_range_month['Day'], obs_tidal_range_month['BD'],'r.',label = "BD_Observation_SIHYMECC")
ax[5].plot(obs_tidal_range_month['Day'], obs_tidal_range_month['BK'],'r.',label = "BK_Observation_SIHYMECC")

Salinity[Salinity.Location==86].plot(x="Date", y="Salinity",ax=ax[7],label="PC Salinity")
Salinity[Salinity.Location==130].plot(x="Date", y="Salinity",ax=ax[7],label="BD Salinity")
Salinity[Salinity.Location==154].plot(x="Date", y="Salinity",ax=ax[7],label="BK Salinity")

ax[0].set_ylabel("Tidal range (m)")
ax[4].set_ylabel("Tidal range (m)")
ax[7].set_ylabel("Salinity")


for i in range(1,9,2):
	ax[i].set_ylabel("Tidal range (m)")
	ax[i].set_xlabel("")
	ax[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
	ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
	ax[i].tick_params(axis='x', rotation=30)
ax[7].set_ylabel("Salinity")


#ax[0].title.set_text('Variation in one station')

# Add Pearson correlation
ax[1].annotate("$R^2=$ {:.2f}; RMSE={:.2f}; Error={:.0f}%".format(pearsonr(compare_model_full[43], compare_model_full.PC)[0],mean_squared_error(compare_model_full[43], compare_model_full.PC,squared=False),100*mean_absolute_percentage_error(compare_model_full[43], compare_model_full.PC)),xy=(0.01, 0.9), xycoords='axes fraction')

ax[3].annotate("$R^2=$ {:.2f}; RMSE={:.2f}; Error={:.0f}%".format(pearsonr(compare_model_full[65], compare_model_full.BD)[0],mean_squared_error(compare_model_full[65], compare_model_full.BD,squared=False),100*mean_absolute_percentage_error(compare_model_full[65], compare_model_full.BD)),xy=(0.01, 0.9), xycoords='axes fraction')

ax[5].annotate("$R^2=$ {:.2f}; RMSE={:.2f}; Error={:.0f}%".format(pearsonr(compare_model_full[77], compare_model_full.BK)[0],mean_squared_error(compare_model_full[77], compare_model_full.BK,squared=False),100*mean_absolute_percentage_error(compare_model_full[77], compare_model_full.BK)),xy=(0.01, 0.9), xycoords='axes fraction')

######Plot Geometry##########################

# Depth
ax[0].set_ylim(-25,0)
ax[0].plot(river_bed.Location,river_bed.Depth,'red')
ax[0].plot(data_river_mean['Location'], data_river_mean['Depth'],'b.')

# Width 
#ax[0].set_ylim(-25,0)
ax[2].plot(river_width.Location,river_width.Width,'red')
ax[2].plot(data_river_mean['Location'], data_river_mean['Width'],'b.')

# Add legend
ax[0].legend(labels=['C-GEM','Observation'])
ax[1].legend(labels=['PC_Simulation','PC_Observation','Observation_SIHYMECC'])

# Add y label
ax[0].set_ylabel('Depth (m)')
ax[2].set_ylabel('Width (m)')

# Set up x, y range, label
for i in range(0,10,2):
	ax[i].set_xlim(-2,210)
	ax[i].axvline(x=141, ls='-', color='k',alpha=0.25, linewidth=1)
	ax[i].axvline(x=130, ls='--', color='k',linewidth=1)
	ax[i].axvline(x=86, ls='--',  color='k',linewidth=1)
	ax[i].axvline(x=156, ls='--',  color='k',linewidth=1)

ax[6].set_xlabel('Distance from reservoir to estuary (km)')

# Add labels for external lines
ax[6].text(156, 0, 'BK station',rotation=90,alpha=0.85)
ax[6].text(141, 0, 'DN River',rotation=90,alpha=0.85)
ax[6].text(130, 0, 'BD station',rotation=90,alpha=0.85)
ax[6].text(86, 0, 'PC station',rotation=90,alpha=0.85)
ax[6].text(0, 0, 'Reservoir',rotation=90,alpha=0.85)


fig.tight_layout()
#fig.subplots_adjust(wspace=0.15, hspace=0)
fig.savefig("OUT\\Check Tidal range %s.jpg" % (filename.strftime("%d-%m-%H-%M")),dpi=200)

# %%
