# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:58:24 2019

@author: bhuban.sarkar
"""
print('STARTING............................')

print('PACKAGE LOADING.....................')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import datetime
import seaborn as sns

#set directory
os.chdir("C:/Users/bhuban.sarkar/OneDrive - TATA INDUSTRIES LIMITED - TATA INSIGHTS & QUANTS DIVISION/Tata Steel - SP#4 Reduction in Solid Fuel/RAW FILE")

columns = ['TIMESTAMP',
'IH_COMBUSTION_AIR_TEMP',
'MACHINE_SPEED',
'SINTER_TEMP_SM_DISCH',
'CO',
'O2',
'COOLER_SPEED',
'TOTAL_BMIX_FLOW',
'TOTAL_CLIME_FLOW',
'TOTAL_LIMESTONE_FLOW',
'TOTAL_RFINES_FLOW',
'TOTAL_SOLID_FUEL_FLOW',
'TEMP_AFTER_ESP',
'WINDBOX_10_TEMP',
                    'WINDBOX_11_TEMP',
                    'WINDBOX_12_TEMP',
                    'WINDBOX_13_TEMP',
                    'WINDBOX_14_TEMP',
                    'WINDBOX_15B_TEMP',
                    'WINDBOX_15A_TEMP',
                    'WINDBOX_16A_TEMP',
                    'WINDBOX_16B_TEMP',
                    'WINDBOX_17A_TEMP',
                    'WINDBOX_17B_TEMP',
                    'WINDBOX_3_TEMP',
                    'WINDBOX_4_TEMP',
                   # 'WINDBOX_5_TEMP',
                    #'WINDBOX_6_TEMP',
                    'WINDBOX_7_TEMP',
                    'WINDBOX_8_TEMP',
                    'WINDBOX_9_TEMP',
                    'WINDBOX_1_TEMP',
                    'WINDBOX_2_TEMP']


SP4_DATA = pd.read_excel('SP4data.xlsx')

SP4_DATA = SP4_DATA[columns]

DATA_all = SP4_DATA

WINDBOX = DATA_all.loc[:,['WINDBOX_10_TEMP',
                    'WINDBOX_11_TEMP',
                    'WINDBOX_12_TEMP',
                    'WINDBOX_13_TEMP',
                    'WINDBOX_14_TEMP',
                    'WINDBOX_15B_TEMP',
                    'WINDBOX_15A_TEMP',
                    'WINDBOX_16A_TEMP',
                    'WINDBOX_16B_TEMP',
                    'WINDBOX_17A_TEMP',
                    'WINDBOX_17B_TEMP',
                    'WINDBOX_3_TEMP',
                    'WINDBOX_4_TEMP',
                    #'WINDBOX_5_TEMP',
                    #'WINDBOX_6_TEMP',
                    'WINDBOX_7_TEMP',
                    'WINDBOX_8_TEMP',
                    'WINDBOX_9_TEMP',
                    'WINDBOX_1_TEMP',
                    'WINDBOX_2_TEMP']]

DATA_all['BTP_TEMP'] = WINDBOX.max(axis=1)

###############################################################################
DATA_all['total_feed_flow'] = DATA_all[['TOTAL_BMIX_FLOW',
                                'TOTAL_CLIME_FLOW',
                                'TOTAL_LIMESTONE_FLOW',
                                'TOTAL_RFINES_FLOW',
                                'TOTAL_SOLID_FUEL_FLOW']].sum(axis=1,skipna = True)

DATA_all['clime_percent'] = (DATA_all['TOTAL_CLIME_FLOW']/DATA_all['TOTAL_BMIX_FLOW'])*100
DATA_all['limestone_percent'] = (DATA_all['TOTAL_LIMESTONE_FLOW']/DATA_all['TOTAL_BMIX_FLOW'])*100
DATA_all['rfines_percent'] = (DATA_all['TOTAL_RFINES_FLOW']/DATA_all['TOTAL_BMIX_FLOW'])*100
DATA_all['sf_percent'] = (DATA_all['TOTAL_SOLID_FUEL_FLOW']/DATA_all['TOTAL_BMIX_FLOW'])*100
DATA_all['flux_percent'] = (DATA_all[['TOTAL_CLIME_FLOW','TOTAL_LIMESTONE_FLOW']].sum(axis=1)/DATA_all['TOTAL_BMIX_FLOW'])*100

#################### Exclude turn up and shut down records ####################
###############################################################################
temp = DATA_all
temp['row_num'] = range(0,len(temp))
temp['row_num'] += 1

temp_zero = temp.loc[temp['MACHINE_SPEED'] <= 0,]
temp_good = temp.loc[temp['MACHINE_SPEED'] > 2.4,]

temp_zero.shape, temp_good.shape #(0, 23), (8465, 23)
temp_good.loc[:,'row_num_next'] = temp_good['row_num'].shift(-1)
temp_good.loc[:,'row_num_next'] = temp_good['row_num_next'].fillna(0).astype(int)
temp_good['zero_flag'] = -99999

zero_flag = []
for i in range(0,len(temp_good)-1):
    print('#####################################')
    #print(i)
    sq = pd.Series(np.arange(temp_good.iloc[i,].row_num,temp_good.iloc[i,].row_num_next,1))
    zero_flag.append(sum(sq.isin(temp_zero['row_num'])))


zero_flag.append(-99999)

temp_good['zero_flag'] =  zero_flag

temp_good_sub = temp_good.loc[temp_good['zero_flag']>0,]

hala = []
for i in range(0,len(temp_good_sub)):
    print('#####################################')
    sq1 = np.arange((temp_good_sub.iloc[i,].row_num + 1),temp_good_sub.iloc[i,].row_num_next,1)
    print(sq1.shape)
    #print(type(sq1))
    hala.extend(sq1)
    #print(hala)

len(hala) #1245
sum(temp_good_sub['row_num_next']-temp_good_sub['row_num']) #1245 + 81(len temp_good_sub)

temp = temp[~temp.row_num.isin(hala)]
temp.shape #(8216, 63)
temp = temp.drop(['row_num'],axis=1)
temp.shape #(9240, 38)

###############################################################################
base_data_MS = temp
base_data_MS.shape #(8216, 62)
base_data_BTP_TEMP = base_data_MS[((base_data_MS.BTP_TEMP >= 350) & (base_data_MS.BTP_TEMP <= 510))]
base_data_BTP_TEMP.shape #(7215,62)
#base_data_TOTAL_CR = base_data_BTP_TEMP[(base_data_BTP_TEMP.CRATE_SP4_GSN > 40)]
#base_data_TOTAL_CR.shape #(7108, 62)

#Missing Value Treatmenet
DATA2 = base_data_BTP_TEMP.dropna(axis=0)
DATA2.shape #(6143, 62)
desc1 = DATA2.describe()
#Outlier Treatment
Float = DATA2.select_dtypes(include = 'float64')
low = .01
high = .99
percentile = Float.quantile([low,high])

LOW_OL = Float.apply(lambda x: x.where(x>percentile.loc[low,x.name],percentile.loc[low,x.name]))
ALL_OL = LOW_OL.apply(lambda x: x.where(x<percentile.loc[high,x.name],percentile.loc[high,x.name]))

raw_data = pd.concat([DATA2.loc[:,['TIMESTAMP']], ALL_OL], axis=1) 

desc2 = raw_data.describe()

raw_data.shape #(8991, 31)

q1 = raw_data['CO'].quantile(0.25)
q3 = raw_data['CO'].quantile(0.75)
IQR = q3-q1

raw_data = raw_data.query('(@q1 - 3*@IQR) <= CO <= (@q3 + 3*@IQR)')

desc3 = raw_data.describe()

###############################################################################

lag = 1 #buffer time #change
raw_data['TIMESTAMP'] = pd.to_datetime(raw_data['TIMESTAMP'])
raw_data['TIMESTAMP_NEXT'] = raw_data['TIMESTAMP'] + datetime.timedelta(hours = lag)

IH_COMB_AIR_TEMP = raw_data[['TIMESTAMP','IH_COMBUSTION_AIR_TEMP']]
IH_COMB_AIR_TEMP.columns = ['TIMESTAMP_NEXT','FUTURE_IH_TEMP']
raw_data = pd.merge(raw_data,IH_COMB_AIR_TEMP, on = 'TIMESTAMP_NEXT', how = 'left')


abs(raw_data.corr()['FUTURE_IH_TEMP']).sort_values()

DATA = raw_data[raw_data['FUTURE_IH_TEMP'] > 0]
DATA.shape #(7403, 40)


###############################################################################
######################################## MODEL ################################

RESPONSE =  ['FUTURE_IH_TEMP']
PREDICTOR = ['SINTER_TEMP_SM_DISCH'
             ,'CO'
             #,'O2'
             #,'IH_COMBUSTION_AIR_FLOW'
             #,'IH_COMBUSTION_AIR_PRESSURE'
             #,'IH_PRESSURE'
             #,'MACHINE_SPEED'
             ,'COOLER_SPEED'
             #,'clime_percent'
             #,'limestone_percent'
             ,'total_feed_flow'
             ,'flux_percent'
             ,'rfines_percent'
             ,'sf_percent'
             #,'RMBBN'
             #,'TEMP_AFTER_ESP'
             #,'CRATE_SP4_GSN'
             #,'CRATE_PSW_RMBBN'
             #,'CRATE_CB_RMBBN'
             #,'CRATE_CB_SP4'
             #,'SUCTION_ESP_OUTLET'
             ,'BTP_TEMP'
              ]

mDATA = DATA[RESPONSE+PREDICTOR]
mDATA.index = DATA['TIMESTAMP']
mDATA = mDATA.dropna(axis=0)    

#chk
corr = mDATA.corr()
plt.figure(figsize=(10,15))
sns.heatmap(corr,fmt=".2f",annot=True,cmap="YlGnBu")
plt.show()
corr_2hours = abs(corr['FUTURE_IH_TEMP']).sort_values()

print(min(mDATA.index))
print(max(mDATA.index))


###############################################################################
#                              TRAIN TEST SPLIT                               #
###############################################################################
train = mDATA[mDATA.index < '2019-07-20 00:00:00'].copy()
test = mDATA[mDATA.index >= '2019-07-20 00:00:00'].copy()
print(train.shape) 
print(test.shape)

###########################SCALING#############################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train[PREDICTOR].values)

X_train_df = train[PREDICTOR] 
X_train = scaler.transform(X_train_df.copy())
y_train_df = train[RESPONSE]

from scipy import stats
y_train,fitted_lambda = stats.boxcox(y_train_df)
scalerY = StandardScaler().fit(y_train)
y_train = scalerY.transform(y_train)

def de_boxcox(np_array,fitted_lambda):
    return np.power((fitted_lambda * np_array + 1 ),(1/fitted_lambda))

#y_train = np.array(y_train_df)
X_test_df = test[PREDICTOR].copy()
X_test = scaler.transform(X_test_df.copy())


JULY = X_test_df[(X_test_df.index >= '2019-07-01 00:00:00')].copy()
scaler_july = StandardScaler().fit(JULY.values)
JULY_test = scaler_july.transform(JULY.copy())


y_test_df = test[RESPONSE]
y_test = scalerY.transform(stats.boxcox(y_test_df , fitted_lambda))

#y_test = np.array(y_test_df)

fig, ax=plt.subplots(1,2)
sns.distplot(y_train, ax=ax[0])
sns.distplot(y_test, ax=ax[1])

print('')
print('*********** DATA SPLIT SHAPE ***********')
print('')
print('Training2 Features Shape:',X_train.shape)
print('Training2 Labels Shape:',y_train.shape)
print('Test Features Shape:',X_test.shape)
print('Test Labels Shape:',y_test.shape)
print('')
print('***************************************')

############################# MODEL FITTING ###################################
from sklearn.linear_model import LinearRegression

# Fit the train data
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


############################ MODEL EVALUATION #################################
train_eva = train[['FUTURE_IH_TEMP']].copy()
test_eva = test[['FUTURE_IH_TEMP']].copy()


train_eva['PRED_SINGLE'] = de_boxcox(scalerY.inverse_transform(regression_model.predict(X_train)),fitted_lambda)
test_eva['PRED_SINGLE'] = de_boxcox(scalerY.inverse_transform(regression_model.predict(X_test)),fitted_lambda)

def MAPE_MAE(df,actual,pred):
    #Calculate MAE & MAPE
    MAE = np.mean(abs(df[actual] - df[pred]))
    MAPE = np.mean(100*(abs(df[actual]-df[pred])/df[actual]))
    return MAPE,MAE 

print('')
print('*********** TRAINING MODEL OUTPUT ***********')
print('')
train_pred_single_MAPE, train_pred_single_MAE = MAPE_MAE(train_eva,'FUTURE_IH_TEMP','PRED_SINGLE')
print('Training Model MAE is:',round(train_pred_single_MAE,2))
print('Training Model MAPE is:',round(train_pred_single_MAPE,2))
print('')
print('*********************************************')
print('')
print('*********** TESTING MODEL OUTPUT ************')
print('')
test_pred_single_MAPE, test_pred_single_MAE = MAPE_MAE(test_eva,'FUTURE_IH_TEMP','PRED_SINGLE')
print('Test Single Model MAE is:',round(test_pred_single_MAE,2))
print('Test Single Model MAPE is:',round(test_pred_single_MAPE,2))
print('')
print('*********************************************')
print('')

###############################################################################
################################# LINEAR MODEL ################################
X_train_LM = pd.DataFrame(X_train,columns=X_train_df.columns,index = X_train_df.index)
import statsmodels.api as sm
#X_train_c =  sm.add_constant(X_train_LM)
X_train_c = X_train_LM.copy()
X_train_c['const'] =  1
sm_model = sm.OLS(y_train,X_train_c).fit()

print_model = sm_model.summary()
print(print_model)

###############################################################################
############################## MODEL EVA GRAPH ################################
test_eva_july = test_eva[(test_eva.index >= '2019-07-20 00:00:00')].copy()
test_eva_july['PRED_SINGLE'] = de_boxcox(scalerY.inverse_transform(regression_model.predict(JULY_test)),fitted_lambda)

test_eva2 = test_eva_july[(test_eva_july.index >= '2019-07-20 00:00:00') &
                     (test_eva_july.index <= '2019-07-24 00:00:00')].copy()
plt.figure(figsize=(15,6))
plt.plot(np.array(range(len(test_eva2))),test_eva2['FUTURE_IH_TEMP'],
         #marker = 'o',markerfacecolor='blue', markersize=8,
         linewidth=3,linestyle = '-', color = 'blue')
plt.plot(np.array(range(len(test_eva2))),test_eva2['PRED_SINGLE'],
         color = 'red',linewidth=1,linestyle = '--',
         marker = 'o',markersize=8,markerfacecolor='yellow'
         )
plt.legend()
plt.show()


MAPE_JULY, MAE_JULY = MAPE_MAE(test_eva2,'FUTURE_IH_TEMP','PRED_SINGLE')

print('')
print('*********************************************')
print('************** USING JULY RANGE *************')
print('')
print('July MAPE :',MAPE_JULY)
print('July MAE :',MAE_JULY)

test_eva1 = test_eva[(test_eva.index >= '2019-07-20 00:00:00') &
                     (test_eva.index <= '2019-07-24 00:00:00')].copy()
#test resul graph
plt.figure(figsize=(15,6))
plt.plot(np.array(range(len(test_eva1))),test_eva1['FUTURE_IH_TEMP'],
         #marker = 'o',markerfacecolor='blue', markersize=8,
         linewidth=3,linestyle = '-', color = 'blue')
plt.plot(np.array(range(len(test_eva1))),test_eva1['PRED_SINGLE'],
         color = 'red',linewidth=1,linestyle = '--',
         marker = 'o',markersize=8,markerfacecolor='yellow'
         )
plt.legend()
plt.show()

MAPE_JUNE, MAE_JUNE = MAPE_MAE(test_eva1,'FUTURE_IH_TEMP','PRED_SINGLE')

print('')
print('*********************************************')
print('**** TESTING ERROR USING OVERALL RANGE ******')
print('')
print('')
print('MAPE :',MAPE_JUNE)
print('MAE :',MAE_JUNE)
##################################################################################
plt.plot(np.array(range(len(X_test_df))),X_test_df['total_feed_flow'],
         #marker = 'o',markerfacecolor='blue', markersize=8,
         linewidth=3,linestyle = '-', color = 'blue')


sns.distplot(mDATA['FUTURE_IH_TEMP'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.show()


JUNE = mDATA[(mDATA.index >= '2019-06-01 00:00:00') &
                     (mDATA.index <= '2019-06-30 00:00:00')].copy()

JULY =  mDATA[(mDATA.index >= '2019-07-01 00:00:00') &
                     (mDATA.index <= '2019-07-30 00:00:00')].copy()


JUNE_DESC = JUNE.describe()
JUNE_DESC.to_excel('JUNE_DESC.xlsx')


JULY_DESC = JULY.describe()
JULY_DESC.to_excel('JULY_DESC.xlsx')




