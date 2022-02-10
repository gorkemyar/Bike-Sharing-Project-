
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import special
# %matplotlib inline
day  = pd.read_csv('/content/sample_data/day.csv')
hour = pd.read_csv('/content/sample_data/hour.csv')

from google.colab import drive
drive.mount('/content/drive')

"""### Basic Attributes of Day.csv

In here, the general outline of the daily data is discovered. It has 731 rows and 16 columns. Dtypes of the data is shown below.
"""

print(day.shape)
print("\n")
print(day.dtypes)
print("\n")
day.head()

"""### Basic Attributes of Hour.csv

In here, the general outline of the hourly data is discovered. It has 17379 rows and 17 columns. Dtypes of the data is shown below.
"""

print(hour.shape)
print("\n")
print(hour.dtypes)
print("\n")
hour.head()

"""### Data Preprocessing

In this part, it is checked if there are any missing values in the data. In addition, validity of both datasets was checked.
"""

print(day.isna().sum(axis=0))


day =day[(day['yr']==0) | (day['yr']==1)]
day =day[(day['season']>=1) | (day['season']<=4)]
day =day[(day['mnth']>=1) | (day['yr']<=12)]
day =day[(day['holiday']==0) | (day['holiday']==1)]
day =day[(day['weekday']>=0) | (day['weekday']<=6)]
day =day[(day['workingday']==0) | (day['workingday']==1)]
day =day[(day['weathersit']>=1) | (day['weathersit']<=4)]
day =day[(day['temp']>=0) | (day['temp']<=1)]
day =day[(day['atemp']>=0) | (day['atemp']<=1)]
day =day[(day['hum']>=0) | (day['hum']<=1)]
day =day[(day['windspeed']>=0) | (day['windspeed']<=1)]
day =day[(day['registered']>=0) & (day['casual']>=0)& (day['cnt']>=0)]
day.shape



print(hour.isna().sum(axis=0))



hour =hour[(hour['yr']==0) | (hour['yr']==1)]
hour =hour[(hour['season']>=1) | (hour['season']<=4)]
hour =hour[(hour['mnth']>=1) | (hour['yr']<=12)]
hour =hour[(hour['holiday']==0) | (hour['holiday']==1)]
hour =hour[(hour['weekday']>=0) | (hour['weekday']<=6)]
hour =hour[(hour['workingday']==0) | (hour['workingday']==1)]
hour =hour[(hour['weathersit']>=1) | (hour['weathersit']<=4)]
hour =hour[(hour['temp']>=0) | (hour['temp']<=1)]
hour =hour[(hour['atemp']>=0) | (hour['atemp']<=1)]
hour =hour[(hour['hum']>=0) | (hour['hum']<=1)]
hour =hour[(hour['windspeed']>=0) | (hour['windspeed']<=1)]
hour =hour[(hour['registered']>=0) & (hour['casual']>=0)& (hour['cnt']>=0)]
hour.shape

"""There is not any invalid Data, though we made the preprocessing for hour.csv.

## Data Exploration

For this part, it is aimed to explore the data and do the necessary hypothesis testing to check our null/alternative hypothesis. Firstly, we checked the correlation between all the features in the data to have a better idea about the data. Then, we moved to check and display the correlations between specific features and registered/casual count. The specific features that we checked the correlation between them, and counts are Temperature, Weather Situation, Wind Speed, Humidity. More specific details about the results of these checks are provided in their respective parts. After that, we checked the distributions of registered and count users in a daily time. Later, as a hypothesis testing, we checked if the count of registered users is evenly distributed across the time. Furthermore, the distribution of casual and registered users according to working days is checked. Lastly, we performed two hypothesis testing. One of them is to check if casual users are more in weekends or not. Other one is to check if registered users are more in working days or not.  Details of these processes are provided in their respective parts.

### Heatmap for Correlartion of Features
"""

names=["holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual","registered","count"]
fig = plt.figure(figsize=(9, 9))
heatmap_data=day.drop(["instant","dteday", "season","yr","mnth"], axis=1).corr(method="pearson")
plt.imshow(heatmap_data.values, cmap="Reds")

plt.clim(-1,1)
plt.colorbar(label="Correlation",fraction=0.1)

plt.yticks(range(len(names)),names)
plt.xticks(range(len(names)),names,rotation=40)
for y in range(heatmap_data.shape[0]):
    for x in range(heatmap_data.shape[1]):       
        plt.text(x , y , '%.2f' % heatmap_data.iloc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center', color='white'
                 )
        
plt.title("Correlation Plot for Features", size=16)
plt.show()


###Temperature vs Counts


plt.figure(figsize=(16,8))
plt.xlabel("Days", size=16)

ax = plt.subplot()
plt.plot(day["dteday"],day["temp"] * 41 * 40,color="black",label="Temperature")
plt.plot(day["dteday"],day["temp"] * 41 *-40,color="black")
ax.bar(day["dteday"], day["registered"], width=1, color="orange", label="Registered")
ax.bar(day["dteday"], day["casual"] * -1, width=1, color="blue", label="Casual")
ax.set_xticks([])
#ax.set_yticks([])
ax.set_ylabel("Counts", size=16)
plt.title("Temperature vs Count", size=16)
plt.legend(loc="upper left")
plt.show()


### Weathersit vs Counts


GR = day.groupby("weathersit")["registered"].mean()
GC = day.groupby("weathersit")["casual"].mean()
df1 = pd.DataFrame(GR)
df2 = pd.DataFrame(GC)

df = pd.merge(df1, df2, on='weathersit', how='outer')

ax = df.plot.bar(color=["SkyBlue","IndianRed"], rot=0, title="Means of Casual & Registered Users According to Weather Situation", figsize=(8,7), width = 0.85)
ax.set_xlabel("weather situation")
ax.set_ylabel("mean")
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(["clear","misty and cloudy","rain/snow","heavy snow or storm"])
plt.xticks(rotation=45)

plt.show()



### Season vs Count


casuals={}
registereds={}
index1 = (day.columns.get_loc("mnth"))
index2 = (day.columns.get_loc("registered"))
index3 = (day.columns.get_loc("casual"))

for i in range(len(day)):
  d          = float(day.iloc[i][index1])
  registered = day.iloc[i][index2]
  casual     = day.iloc[i][index3]
  if d in casuals.keys():
    registereds[d] = registereds[d] + registered
    casuals[d] = casuals[d] + casual
  else:
    registereds[d] = registered
    casuals[d] = casual

casuals2={}
registereds2={}
for i in casuals.keys():
  casuals2[i] = -1*casuals[i]
  registereds2[i] = 1*registereds[i]

plt.figure(figsize=(15,10))
plt.xlabel("Months")
ax = plt.subplot()

colors1 = ["steelblue","skyblue","yellowgreen","olivedrab","darkolivegreen",
           "indianred","lightcoral","salmon","orange","sandybrown","burlywood",
           "lightsteelblue"]
colors2 = ["mediumblue","darkblue","limegreen","green","darkgreen","brown",
           "firebrick","indianred","darkorange","orange","gold","cornflowerblue"]
ax.bar(list(casuals2.keys()), list(casuals2.values()), width=0.9, color=colors1)
ax.bar(list(registereds2.keys()), list(registereds2.values()), width=0.9, color=colors2)
ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.title("Months vs Count", size=16)
ax.axhline(0,color="black")
plt.show()



### Windspeed-Count Density Graph


x = day["windspeed"] 
y = day["registered"]
import plotly.graph_objects as go 
fig= go.Figure(go.Histogram2dContour(x = x,y = y,colorscale = 'Jet',
        contours = dict(showlabels = True,labelfont = dict(family = 'Raleway',color = 'white')),
        hoverlabel = dict(bgcolor = 'white',bordercolor = 'black',font = dict(family = 'Raleway',color = 'black'))), 
        layout=dict(title=dict(text="Windspeed-Count Density Graph for Registered Users"))
)
fig.show()
x = day["windspeed"] 
y = day["casual"]
fig = go.Figure(go.Histogram2dContour(x = x,y = y,colorscale = 'Jet',
        contours = dict(showlabels = True,labelfont = dict(family = 'Raleway',color = 'white')),
        hoverlabel = dict(bgcolor = 'white',bordercolor = 'black',font = dict(family = 'Raleway',color = 'black'))), 
        layout=dict(title=dict(text="Windspeed-Count Density Graph for Casual Users"))
)
fig.show()




### Humidty-Count Density Graph


x = day["hum"] 
y = day["registered"]
import plotly.graph_objects as go 
fig= go.Figure(go.Histogram2dContour(x = x,y = y,
        contours = dict(showlabels = True,labelfont = dict(family = 'Raleway',color = 'white')),
        hoverlabel = dict(bgcolor = 'white',bordercolor = 'black',font = dict(family = 'Raleway',color = 'black'))), 
        layout=dict(title=dict(text="Humidty-Count Density Graph for Registered Users"))
)
fig.show()
x = day["hum"] 
y = day["casual"]
import plotly.graph_objects as go 
fig= go.Figure(go.Histogram2dContour(x = x,y = y,
        contours = dict(showlabels = True,labelfont = dict(family = 'Raleway',color = 'white')),
        hoverlabel = dict(bgcolor = 'white',bordercolor = 'black',font = dict(family = 'Raleway',color = 'black'))), 
        layout=dict(title=dict(text="Humidty-Count Density Graph for Casual Users"))
)
fig.show()



### Distribution of Registered and Casual Users in a Daily Time


groupedR=hour.groupby('hr')['registered'].sum()
groupedC=hour.groupby('hr')['casual'].sum()
fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)  

groupedR.plot.bar(ax=ax[0], color="r")
ax[0].set_title("Registered")

groupedC.plot.bar(ax=ax[1], color="m")
ax[1].set_title("Casual")
#groupedC2.plot(kind="hist", ax=ax[1], bins=20, label="none", color="m", density=True)


plt.suptitle("Hours")
plt.show()


### Hypothesis Testing: Testing whether time slots are evenly distributed or not.

###Null Hypothesis: Count of registered users are evenly distributed across the time (100/24= 4.4--> 4.43 => 13% percent for 3 time slots)<br>
###Alternative Hypothesis: Registered bicycle users cover more than 13% percentage in all times at the beginning and the ending of the working hours (8-9, 17-19) (Null Hypothesis suggests that all times slots are equally distributed. According to that 3 time slots should take 13% percent however alternative hypothesis suggest it should be more than that.)


date=hour.iloc[0]['dteday']
row, column=hour.shape
my=[8,17,18]
sumAll=0
count=0
sumValid=0
for i in range(row):
  if hour.iloc[i]['dteday']==date:
    sumAll+=hour.iloc[i]['registered']
    if hour.iloc[i]['hr'] in my:
      sumValid+=hour.iloc[i]['registered']
  else:
    date=hour.iloc[i]['dteday']
    if sumValid<=sumAll*0.13:
      count+=1
    sumValid=0
    sumAll=0
print("p_value:",count/(row/24))
###We rejected the null hypothesis since p_value <0.05, beginning and end of the working hours cover up more than 13%.



### Distribution of Casual and Registered Users according to Working Days
tm=['Weekend', 'Working Day']
fig, axs=plt.subplots(1,2, figsize=(18, 8),sharey=True)

for i in range(2):
  ax=axs[i]
  indexes=[]
  holiday=day[day['workingday']==i]
  meanCasual=np.median(holiday['casual'])
  meanRegistered=np.median(holiday['registered'])
  if meanCasual>meanRegistered:
    indexes.append('casual')
    indexes.append('registered')
  else:
    indexes.append('registered')
    indexes.append('casual')

  all_values=[]
  for k in indexes:
    temp=[item for item in holiday[k]]
    all_values.append(temp)
  ax.violinplot(all_values, showmedians=True)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.grid()

  ax.set_xticks(np.arange(1, 3))
  ax.set_xticklabels(indexes)
  ax.set_title('Count Distribution for '+tm[i])

fig.suptitle("Count Distribution of Registered-Casual Users", size=20, y=1.04)
plt.tight_layout()
fig.text(0.51, 0.015, 'User Type', va='center', ha='center', size=16)
fig.text(0.001, 0.55, 'Count', rotation='vertical',va='center', ha='center', size=16)
plt.show()



### Hypothesis Testing: Ttest for Casual Users

### Null Hypothesis: Casual users tend to use bicycles regardless of whether the day is a working day or not (50% of usage count is for working day vice versa). <br>
### Alternative Hypothesis: Casual users tend to use more bicycles on the weekends.
npWo=np.array(day[day['workingday']==1]['casual'])
npWe=np.array(day[day['workingday']==0]['casual'])

stats.ttest_ind(npWo, npWe, equal_var=False)
###Since p-value is less than 0.05, the null hypothesis is rejected. According to the statistic value, the mean of weekend is greater than the mean of working day for registered users.


###Hypothesis Testing: Ttest for Registered *Users*
###Null Hypothesis: Registered users uses bicycles regardless of wether the day is a working day or not. <br>
###Alternative Hypothesis: Registered users tend to use bicycles more on weekdays.
npWo=np.array(day[day['workingday']==1]['registered'])
npWe=np.array(day[day['workingday']==0]['registered'])

stats.ttest_ind(npWo, npWe, equal_var=False)

###Since p-value is less than 0.05, the null hypothesis is rejected. According to the statistic value, the mean of working day is greater than the mean of weekend for registered users.




## Machine Learning Models


### Registered User Count: 80% Training, 10% Validation, 10% Test

###In this Machine Learning model, we are implementing multi-linear regression to predict registered user count

from sklearn.model_selection import train_test_split
features=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
## droped dteday since it is not numeric also it is in the data via year, month entries
day_features=day[features].values 
day_registered_label=day['registered'].values

#registered drx stands for day registered x
drx_train, drx_remain, dry_train, dry_remain =train_test_split(day_features, day_registered_label, test_size=0.2, random_state=0)
drx_val, drx_test, dry_val, dry_test= train_test_split(drx_remain, dry_remain, test_size=0.5,random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn import metrics
model_registered = LinearRegression()  
model_registered.fit(drx_train, dry_train)
coeff_df = pd.DataFrame(model_registered.coef_, features, columns=['Coefficient'])  
coeff_df

###Above there are coefficients of the linear equation

from sklearn.metrics import mean_squared_error, mean_absolute_error

dry_Vpred = model_registered.predict(drx_val)

mse = mean_squared_error(dry_val, dry_Vpred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(dry_val, dry_Vpred)
b_mse=mse
b_rsme=rmse ## these will be used for comparing after
b_mae=mae
print("mse: {}".format(mse)) #MSE stands for Mean Squared Error
print("rmse: {}".format(rmse)) #RMSE stands for Root of Mean Squared Error
print("mae: {}".format(mae)) #MAE stands for Mean Absolute Error

#These are the initial values for the errors when we use all of the features.

plt.figure(figsize=(12,8))
plt.plot(dry_Vpred, label="Prediction")
plt.plot(dry_val, label="True")
plt.xlabel("Number of Days")
plt.ylabel("Registered Count")
plt.title("Registered User Count Estimation Graph", size=16)
plt.legend()
plt.show()


feature_main=features
for f in features:
  feature_copy=features.copy()
  feature_copy.remove(f)
  day_features_copy=day[feature_copy].values 
  day_registered_label_copy=day['registered'].values
  drx_train_c, drx_remain_c, dry_train_c, dry_remain_c =train_test_split(day_features_copy, day_registered_label_copy, test_size=0.2, random_state=0)
  drx_val_c, drx_test_c, dry_val_c, dry_test_c= train_test_split(drx_remain_c, dry_remain_c, test_size=0.5,random_state=0)
  model_registered_c = LinearRegression()  
  model_registered_c.fit(drx_train_c, dry_train_c)
  dry_Vpred_c = model_registered_c.predict(drx_val_c)
  mse_c = mean_squared_error(dry_val_c, dry_Vpred_c)
  rmse_c = np.sqrt(mse_c)
  mae_c = mean_absolute_error(dry_val_c, dry_Vpred_c)
  if (rmse_c<rmse) and (mae_c<mae):
    feature_main=feature_copy
    rmse=rmse_c
    mae=mae_c
    mse=mse_c
    model_registered=model_registered_c
    dry_Vpred=dry_Vpred_c
    dry_test=dry_test_c
    dry_val=dry_val_c
    dry_train=dry_train_c
    drx_test=drx_test_c
    drx_val=drx_val_c
    drx_train=drx_train_c

print(features)
print(feature_main) ## without humidty our regression model performs better
print("Before:","mse: {}".format(b_mse),"rmse: {}".format(b_rsme),"mae: {}".format(b_mae))
print("After:","mse: {}".format(mse),"rmse: {}".format(rmse),"mae: {}".format(mae))


def RegisteredUserEstimator(test, drx_val, dry_test, model_registered, drx_test, dry_val):
  if test=="test":
    drx_val=drx_test
    dry_val=dry_test
  dry_pred=model_registered.predict(drx_val)
  plt.figure(figsize=(12,8))
  plt.plot(dry_pred, label="Prediction")
  plt.plot(dry_val, label="True")
  plt.xlabel("Number of Days")
  plt.ylabel("Registered Count")
  plt.title("Registered User Count Estimation Graph", size=16)
  plt.legend()
  plt.show()
RegisteredUserEstimator("val", drx_val, dry_test, model_registered, drx_test, dry_val)


from sklearn.model_selection import train_test_split
features=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
## droped dteday since it is not numeric also in the data via year, month entries
day_features=day[features].values 
day_casual_label=day['casual'].values

# casual dcx stands for day casual x
dcx_train, dcx_remain, dcy_train, dcy_remain =train_test_split(day_features, day_casual_label, test_size=0.2, random_state=0)
dcx_val, dcx_test, dcy_val, dcy_test= train_test_split(dcx_remain, dcy_remain, test_size=0.5,random_state=0)

model_casual = LinearRegression()  
model_casual.fit(dcx_train, dcy_train)
coeff_df = pd.DataFrame(model_casual.coef_, features, columns=['Coefficient'])  
coeff_df


dcy_Vpred = model_casual.predict(dcx_val)
mse = mean_squared_error(dcy_val, dcy_Vpred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(dcy_val, dcy_Vpred)
b_mse=mse
b_rsme=rmse ## these will be used for comparing after
b_mae=mae
print("mse: {}".format(mse))
print("rmse: {}".format(rmse))
print("mae: {}".format(mae))


plt.figure(figsize=(12,8))
plt.plot(dcy_Vpred, label="Prediction")
plt.plot(dcy_val, label="True")
plt.xlabel("Number of Days")
plt.ylabel("Casual Count")
plt.title("Casual User Count Estimation Graph", size=16)
plt.legend()
plt.show()


feature_main=features
for f in features:
  feature_copy=features.copy()
  feature_copy.remove(f)
  day_features_copy=day[feature_copy].values ## droped dteday since it is not numeric also in the data via year, month entries
  day_casual_label_copy=day['casual'].values
  dcx_train_c, dcx_remain_c, dcy_train_c, dcy_remain_c =train_test_split(day_features_copy, day_casual_label_copy, test_size=0.2, random_state=0)
  dcx_val_c, dcx_test_c, dcy_val_c, dcy_test_c= train_test_split(dcx_remain_c, dcy_remain_c, test_size=0.5,random_state=0)
  model_casual_c = LinearRegression()  
  model_casual_c.fit(dcx_train_c, dcy_train_c)
  dcy_Vpred_c = model_casual_c.predict(dcx_val_c)
  mse_c = mean_squared_error(dcy_val_c, dcy_Vpred_c)
  rmse_c = np.sqrt(mse_c)
  mae_c = mean_absolute_error(dry_val_c, dcy_Vpred_c)
  if (rmse_c<rmse) and (mae_c<mae):
    print(f)
    feature_main=feature_copy
    rmse=rmse_c
    mae=mae_c
    mse=mse_c
    model_casual=model_casual_c
    dcy_Vpred=dcy_Vpred_c
    dcy_test=dcy_test_c
    dcy_val=dcy_val_c
    dcy_train=dcy_train_c
    dcx_test=dcx_test_c
    dcx_val=dcx_val_c
    dcx_train=dcx_train_c

print(features)
print(feature_main) ## there is not any change our regression model performs better with all of the features.
print("Before:","mse: {}".format(b_mse),"rmse: {}".format(b_rsme),"mae: {}".format(b_mae))
print("After:","mse: {}".format(mse),"rmse: {}".format(rmse),"mae: {}".format(mae))

def CasualUserEstimator(test, dcx_val, dcy_test, model_casual, dcx_test, dcy_val):
  if test=="test":
    dcx_val=dcx_test
    dcy_val=dcy_test
  dcy_pred=model_casual.predict(dcx_val)
  plt.figure(figsize=(12,8))
  plt.plot(dcy_pred, label="Prediction")
  plt.plot(dcy_val, label="True")
  plt.xlabel("Number of Days")
  plt.ylabel("Casual Count")
  plt.title("Casual User Count Estimation Graph", size=16)
  plt.legend()
  plt.show()
CasualUserEstimator("val", dcx_val, dcy_test, model_casual, dcx_test, dcy_val)


### With KNN

from sklearn.neighbors import KNeighborsRegressor
from matplotlib.colors import ListedColormap
from ipywidgets import interact
myfeatures=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
features = day[myfeatures]
registered_label = day['registered']
casual_label= day['casual']


  #model_casual = KNeighborsClassifier(11, metric='euclidean') 
  #cx_train, cx_remain, cy_train, cy_remain=train_test_split(features, casual_label, test_size=0.2, random_state=0)
  #cx_val, cx_test, cy_val, cy_test=train_test_split(cx_remain, cy_remain, test_size=0.5, random_state=0) 
  #model_casual.fit(cx_train, cy_train)
best_k=0
mse_Rknn = 10000000000
rmse_Rknn = 10000000000
mae_Rknn= 100000000000 ## sufficiently big numbers for start
for k in range(1,20):

  model_registered2 = KNeighborsRegressor(k, metric='euclidean')

  rx_train, rx_remain, ry_train, ry_remain=train_test_split(features, registered_label, test_size=0.2, random_state=0)
  rx_val, rx_test, ry_val, ry_test=train_test_split(rx_remain, ry_remain, test_size=0.5, random_state=0) 
  model_registered2.fit(rx_train, ry_train)
  
  
  rx_Vpred=model_registered2.predict(rx_val)
  #print(rx_Vpred)
  mse_Rknn_new = mean_squared_error(rx_Vpred, ry_val)
  rmse_Rknn_new = np.sqrt(mse_Rknn)
  mae_Rknn_new = mean_absolute_error(ry_val, rx_Vpred)

  if (mse_Rknn_new<mse_Rknn) and (rmse_Rknn_new<rmse_Rknn) and (mae_Rknn_new<mae_Rknn):
    mse_Rknn=mse_Rknn_new
    rmse_Rknn=rmse_Rknn_new
    mae_Rknn=mae_Rknn_new
    best_k=k

print("mse: {}".format(mse_Rknn))
print("rmse: {}".format(rmse_Rknn))
print("mae: {}".format(mae_Rknn))

model_registered2 = KNeighborsRegressor(best_k, metric='euclidean')
rx_train, rx_remain, ry_train, ry_remain=train_test_split(features, registered_label, test_size=0.2, random_state=0)
rx_val, rx_test, ry_val, ry_test=train_test_split(rx_remain, ry_remain, test_size=0.5, random_state=0) 
model_registered2.fit(rx_train, ry_train)
rx_Vpred=model_registered2.predict(rx_val)

plt.figure(figsize=(12,8))
plt.plot(rx_Vpred, label="Prediction")
plt.plot(np.array(ry_val), label="True")
plt.xlabel("Number of Days")
plt.ylabel("Registered Count")
plt.title("Registered User Count Estimation With KNN", size=16)
plt.legend()
plt.show()

best_k=0
mse_Rknn = 10000000000
rmse_Rknn = 10000000000
mae_Rknn= 100000000000 ## sufficiently big numbers for start
for k in range(1,20):

  model_casual2 = KNeighborsRegressor(11, metric='euclidean') 
  cx_train, cx_remain, cy_train, cy_remain=train_test_split(features, casual_label, test_size=0.2, random_state=0)
  cx_val, cx_test, cy_val, cy_test=train_test_split(cx_remain, cy_remain, test_size=0.5, random_state=0) 
  model_casual2.fit(cx_train, cy_train)
  
  
  cx_Vpred=model_casual2.predict(cx_val)
  #print(rx_Vpred)
  mse_Rknn_new = mean_squared_error(cx_Vpred, cy_val)
  rmse_Rknn_new = np.sqrt(mse_Rknn)
  mae_Rknn_new = mean_absolute_error(cy_val, cx_Vpred)

  if (mse_Rknn_new<mse_Rknn) and (rmse_Rknn_new<rmse_Rknn) and (mae_Rknn_new<mae_Rknn):
    mse_Rknn=mse_Rknn_new
    rmse_Rknn=rmse_Rknn_new
    mae_Rknn=mae_Rknn_new
    best_k=k

print("mse: {}".format(mse_Rknn))
print("rmse: {}".format(rmse_Rknn))
print("mae: {}".format(mae_Rknn))

model_casual2 = KNeighborsRegressor(best_k, metric='euclidean')
cx_train, cx_remain, cy_train, cy_remain=train_test_split(features, registered_label, test_size=0.2, random_state=0)
cx_val, cx_test, cy_val, cy_test=train_test_split(cx_remain, cy_remain, test_size=0.5, random_state=0) 
model_casual2.fit(cx_train, cy_train)
cx_Vpred=model_casual2.predict(cx_val)

plt.figure(figsize=(12,8))
plt.plot(cx_Vpred, label="Prediction")
plt.plot(np.array(cy_val), label="True")
plt.xlabel("Number of Days")
plt.ylabel("Casual Count")
plt.title("Casual User Count Estimation With KNN", size=16)
plt.legend()
plt.show()

def HourlyUserEstimation(myTest):
  features=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
  ## droped dteday since it is not numeric also in the data via year, month entries
  myhours=[8,12,14,17,18]
  fig, axs=plt.subplots(5,2, figsize=(20,15))
  for h in range(len(myhours)):
    hour_new=hour[hour['hr']==myhours[h]]
    hour_features=hour_new[features].values 
    hour_registered_label=hour_new['registered'].values
    hour_casual_label=hour_new['casual'].values
    #registered drx stands for day registered x
    hrx_train, hrx_remain, hry_train, hry_remain =train_test_split(hour_features, hour_registered_label, test_size=0.2, random_state=0)
    hrx_val, hrx_test, hry_val, hry_test= train_test_split(hrx_remain, hry_remain, test_size=0.5,random_state=0)

    hcx_train, hcx_remain, hcy_train, hcy_remain =train_test_split(hour_features, hour_casual_label, test_size=0.2, random_state=0)
    hcx_val, hcx_test, hcy_val, hcy_test= train_test_split(hcx_remain, hcy_remain, test_size=0.5,random_state=0)

    model_registered_hour = LinearRegression()  
    model_registered_hour.fit(hrx_train, hry_train)

    model_casual_hour = LinearRegression()  
    model_casual_hour.fit(hcx_train, hcy_train)
    if myTest=="test":
      hcx_val=hcx_test
      hcy_val=hcy_test
      hrx_val=hrx_test
      hry_val=hry_test
    hcx_val_pred=model_casual_hour.predict(hcx_val)
    hrx_val_pred=model_registered_hour.predict(hrx_val)

    
    axs[h,0].plot(hcx_val_pred, label="Prediction")
    axs[h,0].plot(hcy_val, label= "True")
    atitle="Casual, Time is "+ str(myhours[h])
    axs[h,0].set_title(atitle)
    axs[h,0].legend(loc="lower left")
    axs[h,1].plot(hrx_val_pred,label="Prediction")
    axs[h,1].plot(hry_val, label="True")
    atitle="Registered, Time is "+ str(myhours[h])
    axs[h,1].set_title(atitle)
    axs[h,1].legend(loc="lower left")
  plt.suptitle("Daily Hours", size=20)
  
  plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.1,hspace=0.4)
  plt.show()

HourlyUserEstimation("val")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve
#manuel encoding
def change(a):
  if a==True:
    return 1
  return 0
hour['label']=(hour['registered']>hour['casual']) ## creating a new label that compares registered and casual count
hour['label']=hour['label'].apply(change)
mydict=hour['label'].value_counts().to_dict()
features=['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
x=hour[features]
y=hour['label']
Ratio=mydict[1]/mydict[0]

x_train, x_remain, y_train, y_remain = train_test_split(x,y, test_size=0.2, random_state=0)
x_val, x_test, y_val, y_test =train_test_split(x_remain,y_remain, test_size=0.5, random_state=0)

auprc=[]
n_estimators= [50,100,200,300,500]
for n in n_estimators:
  new_model=RandomForestClassifier(n_estimators=n, class_weight={0:Ratio, 1:1},random_state=0)
  new_model.fit(x_train, y_train)
  precision, recall, threshold =precision_recall_curve(y_val, new_model.predict_proba(x_val)[:, 1])
  auprc.append(auc(recall, precision))

maxiPRC=max(auprc)
idxmaxiPrc=auprc.index(maxiPRC)
max_N=n_estimators[idxmaxiPrc]
max_features= [3,4,5,6,7,8,10]
auprc2=[]
for m in max_features:
  new_model=RandomForestClassifier(n_estimators=max_N, class_weight={0:Ratio, 1:1},max_features=m, random_state=0)
  new_model.fit(x_train, y_train)
  precision, recall, _ =precision_recall_curve(y_val, new_model.predict_proba(x_val)[:,1])
  auprc2.append(auc(recall, precision))
max_M=max_features[auprc2.index(max(auprc2))]
print("N estimator is ", max_N, ", Max feature is ", max_M,".", sep="")

model_rf=RandomForestClassifier(n_estimators=max_N, class_weight={0:Ratio, 1:1}, max_features=max_M, random_state=0)
model_rf.fit(x_train,y_train)

from sklearn.metrics import RocCurveDisplay

def WhoUsedMost(test,model_rf, x_val, x_test, y_val,y_test):

  fig, ax=plt.subplots()
  if test=="test":
    RocCurveDisplay.from_estimator(model_rf, x_test, y_test, ax=ax)
  else:  
    RocCurveDisplay.from_estimator(model_rf, x_val, y_val, ax=ax)
  ax.set_title("Receiver Operating Characteristics")
  ax.set_xlabel("False Positive Rate")
  ax.set_ylabel("True Positive Rate")
  ax.set_xlim(left = 0)
  ax.set_ylim(bottom = 0)
  for d in ax.spines:
    ax.spines[d].set_visible(False)
  tmp=np.linspace(0,1,6, endpoint=True)
  plt.plot(tmp,tmp, linestyle="--")

  plt.show()

WhoUsedMost("val",model_rf, x_val, x_test, y_val,y_test)


### Working Day Estimation



mydict=hour['workingday'].value_counts().to_dict()
features=['mnth', 'registered','casual', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
x=hour[features]
y=hour['workingday']
Ratio=mydict[1]/mydict[0]

x_train2, x_remain2, y_train2, y_remain2 = train_test_split(x,y, test_size=0.2, random_state=0)
x_val2, x_test2, y_val2, y_test2 =train_test_split(x_remain2,y_remain2, test_size=0.5, random_state=0)

auprc=[]
n_estimators= [50,100,200,300,500]
for n in n_estimators:
  new_model=RandomForestClassifier(n_estimators=n, class_weight={0:Ratio, 1:1},random_state=0)
  new_model.fit(x_train2, y_train2)
  precision, recall, threshold =precision_recall_curve(y_val, new_model.predict_proba(x_val2)[:, 1])
  auprc.append(auc(recall, precision))

maxiPRC=max(auprc)
idxmaxiPrc=auprc.index(maxiPRC)
max_N=n_estimators[idxmaxiPrc]
max_features= [3,4,5,6,7,8]
auprc2=[]
for m in max_features:
  new_model=RandomForestClassifier(n_estimators=max_N, class_weight={0:Ratio, 1:1},max_features=m, random_state=0)
  new_model.fit(x_train2, y_train2)
  precision, recall, _ =precision_recall_curve(y_val2, new_model.predict_proba(x_val2)[:,1])
  auprc2.append(auc(recall, precision))
max_M=max_features[auprc2.index(max(auprc2))]
print("N estimator is ", max_N, ", Max feature is ", max_M,".", sep="")

model_rf2=RandomForestClassifier(n_estimators=max_N, class_weight={0:Ratio, 1:1}, max_features=max_M, random_state=0)
model_rf2.fit(x_train2,y_train2)

from sklearn.metrics import RocCurveDisplay

def WorkingDayEstimator(test,model_rf2, x_val2, x_test2, y_val2,y_test2):

  fig, ax=plt.subplots()
  if test=="test":
    RocCurveDisplay.from_estimator(model_rf2, x_test2, y_test2, ax=ax)
  else:  
    RocCurveDisplay.from_estimator(model_rf2, x_val2, y_val2, ax=ax)
  ax.set_title("Receiver Operating Characteristics")
  ax.set_xlabel("False Positive Rate")
  ax.set_ylabel("True Positive Rate")
  ax.set_xlim(left = 0)
  ax.set_ylim(bottom = 0)
  for d in ax.spines:
    ax.spines[d].set_visible(False)
  tmp=np.linspace(0,1,6, endpoint=True)
  plt.plot(tmp,tmp, linestyle="--")

  plt.show()

WorkingDayEstimator("val",model_rf2, x_val2, x_test2, y_val2,y_test2)



### Results & Discussion



###Results for Registered Users
RegisteredUserEstimator("test", drx_val, dry_test, model_registered, drx_test, dry_val)



###Results for Casual Users
CasualUserEstimator("test", dcx_val, dcy_test, model_casual, dcx_test, dcy_val)


### Results for KNN
rx_tpred=model_registered2.predict(rx_test)

plt.figure(figsize=(12,8))
plt.plot(rx_tpred, label="Prediction")
plt.plot(np.array(ry_test), label="True")
plt.xlabel("Number of Days")
plt.ylabel("Registered Count")
plt.title("Registered User Count Estimation With KNN", size=16)
plt.legend()
plt.show()

##Casual test cases for KNN
cx_tpred=model_casual2.predict(cx_test)
plt.figure(figsize=(12,8))
plt.plot(cx_tpred, label="Prediction")
plt.plot(np.array(cy_test), label="True")
plt.xlabel("Number of Days")
plt.ylabel("Casual Count")
plt.title("Casual User Count Estimation With KNN", size=16)
plt.legend()
plt.show()

### Hourly User Estimation Test Part
HourlyUserEstimation("test")


###Who Used Most Test Part
WhoUsedMost("test",model_rf, x_val, x_test, y_val,y_test)



###Working Day Estimation Test Part
WorkingDayEstimator("test",model_rf2, x_val2, x_test2, y_val2,y_test2)
