# -*- coding: utf-8 -*-
"""Feature Engineering and Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tyvZ6wS0ZFAJ_eg3wRmaQeJE5BoM8Wir

# Data Preparation

# Data Preprocessing and Feature Engineering
"""

from google.colab import drive

drive.mount('/content/gdrive')

log_dir = "/content/gdrive/My Drive/workload_predictor_vm"

# Commented out IPython magic to ensure Python compatibility.
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pandas import read_csv, datetime
from pandas.plotting import autocorrelation_plot
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from sklearn.linear_model import LassoCV, RidgeCV
from itertools import product
from tqdm import tqdm_notebook
import matplotlib.dates as mdates
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Import required packages
import os
import glob
import pandas as pd

# Define a function to load and concatenate files from a given directory
def load_and_concatenate(path):
    # Get a list of all .csv files in the specified directory
    all_files = glob.glob(os.path.join(path, "*.csv"))

    # For each file, read it into a DataFrame and add a new column 'VM'
    # that contains the filename (without the .csv extension)
    df_from_each_file = (pd.read_csv(f, sep=';\t').assign(VM=os.path.basename(f).split('.')[0]) for f in all_files)

    # Concatenate all the DataFrames together
    concatenated_df = pd.concat(df_from_each_file)

    return concatenated_df

# Load and concatenate data from the directories for July, August, and September
df_july = load_and_concatenate(log_dir + '/2013-7/')
df_august = load_and_concatenate(log_dir + '/2013-8/')
df_september = load_and_concatenate(log_dir + '/2013-9/')

# Combine all three months of data
concatenated_df = pd.concat([df_july, df_august, df_september])

# Display the first few rows of the DataFrame
concatenated_df.head()

# path = log_dir + '/2013-7/'
# all_files = glob.glob(os.path.join(path, "*.csv"))
# df_from_each_file = (pd.read_csv(f, sep = ';\t').assign(VM=os.path.basename(f).split('.')[0]) for f in all_files)
# concatenated_df = pd.concat(df_from_each_file)

# path =  log_dir +'/2013-8/'
# all_files = glob.glob(os.path.join(path, "*.csv"))
# df_from_each_file = (pd.read_csv(f, sep = ';\t').assign(VM=os.path.basename(f).split('.')[0]) for f in all_files)
# concatenated_df8   = pd.concat(df_from_each_file)

# path =  log_dir +'/2013-9/'
# all_files = glob.glob(os.path.join(path, "*.csv"))
# df_from_each_file = (pd.read_csv(f, sep = ';\t').assign(VM=os.path.basename(f).split('.')[0]) for f in all_files)
# concatenated_df9   = pd.concat(df_from_each_file)

# newdat = concatenated_df.append(concatenated_df8)
# newerdat = newdat.append(concatenated_df9)
# concatenated_df = newerdat

concatenated_df.head()

concatenated_df.to_csv(log_dir + '/concatenated_df.csv', index=False)

"""Read the CSV directly if its already there in the VM

"""

concatenated_df = pd.read_csv(log_dir + '/concatenated_df.csv')

concatenated_df['Timestamp'] = pd.to_datetime(concatenated_df['Timestamp [ms]'], unit = 's')
concatenated_df.apply(pd.to_numeric, errors='ignore')

# Date Feature Engineering
concatenated_df['weekday'] = concatenated_df['Timestamp'].dt.dayofweek
concatenated_df['weekend'] = ((concatenated_df.weekday) // 5 == 1).astype(float)
concatenated_df['month']=concatenated_df.Timestamp.dt.month
concatenated_df['day']=concatenated_df.Timestamp.dt.day
concatenated_df.set_index('Timestamp',inplace=True)

# Other Feature Engineering
concatenated_df["CPU usage prev"] = concatenated_df['CPU usage [%]'].shift(1)
concatenated_df["CPU_diff"] = concatenated_df['CPU usage [%]'] - concatenated_df["CPU usage prev"]
concatenated_df["received_prev"] = concatenated_df['Network received throughput [KB/s]'].shift(1)
concatenated_df["received_diff"] = concatenated_df['Network received throughput [KB/s]']- concatenated_df["received_prev"]
concatenated_df["transmitted_prev"] = concatenated_df['Network transmitted throughput [KB/s]'].shift(1)
concatenated_df["transmitted_diff"] = concatenated_df['Network transmitted throughput [KB/s]']- concatenated_df["transmitted_prev"]

concatenated_df = concatenated_df.fillna(method='ffill')

concatenated_df.shape

concatenated_df.head()

"""Optional step, storing for quick loading if required later."""

# concatenated_df.to_csv(log_dir + '/df_feature.csv', index=False)

# concatenated_df = pd.read_csv(log_dir + '/df_feature.csv')

hourlydat = concatenated_df.resample('H').sum()

hourlydat.to_csv(log_dir + '/df_scaled.csv')
hourlydat.head()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set the seaborn style
sns.set_style("whitegrid")

plt.figure(figsize=(12,6))
pd.plotting.autocorrelation_plot(hourlydat['CPU usage [MHZ]'])

# Add necessary titles and labels
plt.title('Autocorrelation of CPU Usage')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.savefig(log_dir + '/ac_cpu_usage.png')
# Show the plot
plt.show()

"""# CPU Capacity Provisioning and Usage Analysis"""

overprovision = pd.DataFrame(hourlydat['CPU usage [MHZ]'])
overprovision['CPU capacity provisioned'] = pd.DataFrame(hourlydat['CPU capacity provisioned [MHZ]'])

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Apply seaborn styles
sns.set_style("whitegrid")

# Create a larger figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the CPU usage
sns.lineplot(x=overprovision.index, y='CPU usage [MHZ]', data=overprovision, ax=ax, label='CPU usage [MHZ]', color='steelblue', linewidth=2.5)

# Plot the CPU capacity provisioned
sns.lineplot(x=overprovision.index, y='CPU capacity provisioned', data=overprovision, ax=ax, label='CPU capacity provisioned [MHZ]', color='tomato', linewidth=2.5)

# Set titles and labels
ax.set_title('CPU Capacity and Usage Comparison')
ax.set_ylabel((r'CPU [MHz]  $e^{7}$'))
ax.set_xlabel('Date')

ax.legend(loc='best')

# Set the font size of the tick labels
# ax.tick_params(labelsize=1)

# Format the y-axis
ax.ticklabel_format(axis='y', style='sci', scilimits=(1,6))

# Save and show the figure
plt.savefig(log_dir + '/cpu_over_under.png')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Apply seaborn styles
sns.set_style("whitegrid")

# Create a larger figure
fig, ax = plt.subplots(figsize=(12, 6))

# Filter data based on date
# overprovision_filtered = overprovision[(overprovision.index > '2013-08-31 23:59:00') & (overprovision.index < '2013-10-01 0:00:00')]
overprovision_filtered = overprovision
# Calculate median of cpu provisioned
cpu_provisioned_median = overprovision_filtered['CPU capacity provisioned'].median()
cpu_usage_median = overprovision_filtered['CPU usage [MHZ]'].median()

# Calculate 95th percentile of CPU usage
cpu_usage_95th = overprovision_filtered['CPU usage [MHZ]'].quantile(0.95)

print(cpu_usage_median, cpu_provisioned_median, cpu_usage_95th)

# Plot CPU usage
sns.lineplot(x=overprovision_filtered.index, y='CPU usage [MHZ]', data=overprovision_filtered, ax=ax, color='darkslategray', linewidth=2.5, label='CPU usage [MHZ]')

# Plot CPU capacity provisioned
sns.lineplot(x=overprovision_filtered.index, y='CPU capacity provisioned', data=overprovision_filtered, ax=ax, color='darkorange', linewidth=2.5, label='CPU capacity provisioned')

# Set titles and labels
ax.set_title('CPU Usage[MHz] & CPU Provisioned [MHz] Comparison')
ax.set_ylabel((r'CPU [MHz]  $e^{7}$'))
ax.set_xlabel('Date')

# Add median lines
plt.axhline(y=cpu_provisioned_median, color='navy', linestyle='--', label='CPU provisioned median', linewidth=2.5)

# Add 95th percentile line
plt.axhline(y=cpu_usage_95th, color='darkgreen', linestyle='--', label='CPU usage 95th percentile', linewidth=2.5)

# Adjust legend and axis
ax.legend(loc='best')
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (1,6))

# Save and show the figure
plt.savefig(log_dir + '/CPU_cap_under_1mo.png')
plt.tight_layout()
plt.show()

"""**Detailed Interpretation of the Analysis:**

From our analysis, we can infer the following:

1. **Median CPU usage**: The median value of CPU usage during the specified period ('2013-08-31 23:59:00' to '2013-10-01 0:00:00') is approximately 4,554,176 MHz. This statistic is the middle value of the CPU usage that separates the higher half from the lower half of the data set. Essentially, this indicates that half of the observed CPU usage values were below this level, and half were above this level.

2. **Median CPU capacity provisioned**: The median value of the CPU capacity provisioned during the same period is considerably larger, around 45,484,316 MHz. This is the mid-point value of the provisioned CPU capacity, suggesting that the typical provisioned CPU capacity is much higher than the typical CPU usage.

3. **95th percentile of CPU usage**: The value at the 95th percentile of the CPU usage is approximately 11,136,863 MHz. The 95th percentile is a statistical measure indicating that 95% of the observed CPU usage values were less than or equal to this value. In contrast, in only 5% of the instances was the CPU usage higher than this value.

From a capacity planning perspective, these observations suggest potential overprovisioning of CPU capacity. The median provisioned capacity is significantly larger than the median usage, indicating that most of the time, the provisioned CPU capacity is not fully utilized, potentially leading to wastage of resources.

However, considering the 95th percentile of CPU usage provides a more nuanced view. If we provision capacity based on this value, we would be able to handle the peak CPU usage for 95% of the time, potentially achieving a more cost-effective balance between resource utilization and capacity provision.

Nevertheless, it's crucial to ensure that the system can handle instances where usage exceeds this level, which account for 5% of the time. To mitigate risks associated with such instances, additional strategies like flexible scaling could be considered.


"""

# hourlytransmit = hourlydat['Network transmitted throughput [KB/s]']
# hourlytransmit.plot(color = "purple",linewidth = 4,  figsize=(10, 5))
# plt.title('Transmitted Throughput [KB/s] Totals \n Resampled & Aggregated Hourly',fontsize=15);
# plt.ylabel('Transmitted Throughput [KB/s]', fontsize=15);
# plt.xlabel('', fontsize=15);
# plt.show()

# hourlyreceive = hourlydat['Network received throughput [KB/s]']
# hourlyreceive.plot( linewidth = 4, figsize=(10, 5))
# plt.title('Received Throughput [KB/s] Totals \n Resampled & Aggregated Hourly',fontsize=15);
# plt.ylabel('Received Throughput [KB/s]', fontsize=15);
# plt.xlabel('', fontsize=15);
# plt.yticks(fontsize=15);
# plt.xticks(fontsize=15);
# plt.show()

# hourlyprov = hourlydat['CPU capacity provisioned [MHZ]']
# hourlyprov.plot(color = "g", linewidth = 4, figsize=(10, 5))
# plt.title('CPU Provisioned Totals \n Resampled & Aggregated Hourly',fontsize=15);
# plt.ylabel('CPU Capacity Provisioned [MHz]  $e^{7}$', fontsize=15);
# plt.xlabel('', fontsize=15);
# plt.yticks(fontsize=15);
# plt.xticks(fontsize=15);
# plt.show()

# hourlycpu = hourlydat['CPU usage [MHZ]']
# hourlycpu.plot(linewidth = 4, figsize=(10, 5))
# plt.title('CPU Usage Totals \n Resampled & Aggregated Hourly',fontsize=15);
# plt.ylabel('CPU usage [MHz]   $e^{7}$', fontsize=15);
# plt.xlabel('', fontsize=15);
# plt.yticks(fontsize=15);
# plt.xticks(fontsize=15);
# plt.show()

"""# Augmented Dickey-Fuller Test for Stationarity

The Augmented Dickey-Fuller (ADF) test is a type of statistical test called a unit root test. The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.

The null hypothesis of the ADF test is that the time series is not stationary (has some time-dependent structure). The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.

- **Null Hypothesis (H0)**: If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
- **Alternate Hypothesis (H1)**: The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%) suggests we should reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).

- **p-value > 0.05**: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
- **p-value <= 0.05**: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

## Results
Our ADF test gives a p-value of 0.000008, which is less than 0.05, and as such we can reject the null hypothesis. There is strong evidence against the null hypothesis, hence we conclude that the time series is stationary.

Furthermore, the ADF statistic of -5.213157 is less than the value of -3.439891 at 1%. This confirms that we can reject the null hypothesis with a 99% confidence level.

This implies that our 'CPU usage [MHZ]' series does not have a unit root and, in turn, that it is stationary. This is important as the assumption of stationarity is necessary in order to use models like ARIMA for forecasting.

This suggests we don't need to apply any differencing to make the series stationary. However, the series could still have trend or seasonality. A detailed analysis of residuals after fitting a model would be required to ensure these components have been adequately addressed.
"""

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(overprovision['CPU usage [MHZ]'])

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set the seaborn style
sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6))
plt.subplots_adjust(hspace = 0.5)
# Plot the ACF on ax1
plot_acf(overprovision['CPU usage [MHZ]'], lags=20, zero=False, ax=ax1)

# Add labels to the plot
ax1.set_title('Autocorrelation Function (ACF)')
ax1.set_xlabel('Lags')
ax1.set_ylabel('Correlation')

# Plot the PACF on ax2
plot_pacf(overprovision['CPU usage [MHZ]'], lags=20, zero=False, ax=ax2)

# Add labels to the plot
ax2.set_title('Partial Autocorrelation Function (PACF)')
ax2.set_xlabel('Lags')
ax2.set_ylabel('Correlation')

# Save the figure
plt.savefig(log_dir + '/acf_pacf.png')
# Show the plot
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    # Reducing the sample size for clarity, change the value as per your requirement
    timeseries = timeseries.resample('D').mean()

    # Determining rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Plotting rolling statistics
    plt.figure(figsize=(14,7))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig(log_dir+'/adf_test.png')
    plt.show(block=False)

    # Performing Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(overprovision['CPU usage [MHZ]'])

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # Transform the Series into DataFrame for a better visual output
    dfoutput = pd.DataFrame(dfoutput).T
    return dfoutput

dfoutput = test_stationarity(overprovision['CPU usage [MHZ]'])
print(dfoutput)

"""# Seasonality and Trend Analysis

Upon analyzing our time series data with a seasonal decomposition, we notice some distinct patterns in the seasonality and trend components of our data.

## Seasonality

Our seasonal component fluctuates between -1 and 1 with a regular, repeating pattern. This suggests that there is a significant seasonal effect in our data. In the context of our CPU usage data, this could potentially indicate that there are certain times of day, week, or year when CPU usage predictably increases or decreases.

The consistent pattern suggests that this seasonality is stable over time - that is, the timing and magnitude of the seasonal fluctuations do not change. This is a useful property for forecasting, as we can use the seasonal component to make predictions about future data points.

However, it's important to note that while the magnitude of our seasonal fluctuations is relatively small (ranging from -1 to 1), the impact of these fluctuations on our forecasts could be significant, especially if our overall CPU usage values are also small. Thus, it will be important for our chosen forecasting model to capture this seasonality.

## Trend

The trend component of our data appears to be quite random, showing a varying pattern that doesn't have a clear direction. This randomness in the trend component could mean a couple of things:

1. **No Trend**: If the variations in the trend component are just random noise, this could indicate that there is no underlying trend in our data. In other words, aside from the seasonal fluctuations, the CPU usage does not consistently increase or decrease over time.

2. **Non-Linear or Complex Trend**: Alternatively, if the variations are more than just random noise, this could indicate that there is a non-linear or complex trend in our data that isn't being captured by the seasonal decomposition. Non-linear trends could include quadratic or exponential trends, while complex trends could involve multiple interacting factors.

In either case, the apparent randomness of the trend component suggests that our data is relatively stationary, and does not have a strong linear trend. This is consistent with the results of our Dickey-Fuller test, which also suggested that our data is stationary.

However, it's worth noting that many time series forecasting models (like ARIMA) are designed to work best with data that has a linear trend. If our data has a non-linear or complex trend, we may need to consider different models or transformations of our data to effectively capture this trend.

In summary, our analysis suggests that while our CPU usage data is largely stationary and exhibits significant seasonality, there may be complexities in the trend component that we will need to consider when selecting and fitting a forecasting model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Apply seasonal decomposition
decomposition = seasonal_decompose(overprovision['CPU usage [MHZ]'])

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Set the overall theme for the plot
sns.set_style("whitegrid")

# Create the figure and subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 15))

# Plot the original data
axes[0].plot(overprovision['CPU usage [MHZ]'], color='blue')
axes[0].set_title('Original CPU Usage Data')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('CPU Usage [MHZ]')

# Plot the trend component
axes[1].plot(trend, color='purple')
axes[1].set_title('Trend Component of CPU Usage')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('CPU Usage [MHZ]')

# Plot the seasonality component
axes[2].plot(seasonal, color='green')
axes[2].set_title('Seasonality Component of CPU Usage')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Seasonality')

# Plot the residuals
axes[3].plot(residual, color='red')
axes[3].set_title('Residuals of CPU Usage')
axes[3].set_xlabel('Date')
axes[3].set_ylabel('Residuals')

# Set the layout to be tight for aesthetic purposes
fig.tight_layout()

plt.savefig(log_dir + '/seasonality_analysis.png')
# Display the plot
plt.show()

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.style.use('seaborn-white')
    plt.figure(figsize=(12,6))
    plt.title("Moving Average (window size = {})".format(window))
    plt.ylabel('CPU usage [MHz]   $e^{7}$');
    plt.xlabel('Time');
    plt.yticks();
    plt.xticks();
    plt.plot(rolling_mean, "purple", label="Rolling Mean Trend",linewidth = 5)

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper/Lower Bound", linewidth = 1)
        plt.plot(lower_bond, "r--", linewidth = 1)

        # find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10, label = "Anomalies")

    plt.plot(series[window:], label="Actual CPU Usage", linewidth = 1)
    plt.legend(loc="upper right")
    plt.savefig(log_dir + '/anomaly_detection.png')
    plt.grid(True)

plotMovingAverage(hourlydat[['CPU usage [MHZ]']], 24, plot_intervals=True, plot_anomalies=True)
