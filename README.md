<h1 align="center">CloudAIBus Framework</h1>
<div align="center">
  <a href="https://github.com/your-repo/CloudAIBus/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
  <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
  <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyour-repo%2FCloudAIBus&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
  
  <a href="https://doi.org/10.1007/s10586-024-04562-9">
    <img src="https://img.shields.io/badge/DOI-10.1007%2Fs10586-024-04562-9-green.svg" alt="DOI">
  </a>
</div>

CloudAIBus is an AI-based testbed for cloud computing environments, focusing on effective resource allocation using deep learning models. The framework leverages the DeepAR model for robust CPU usage forecasting to make cost-effective resource allocation decisions. It was evaluated using Google Colab and AWS SageMaker, showing superior performance compared to LSTM and ARIMA-based models in terms of MAE, MAPE, and MSE.

<!-- <img src="https://github.com/your-repo/CloudAIBus/blob/master/wiki/CloudAIBus.jpg" width="900" align="middle"> -->
## Advantages of CloudAIBus
1. Efficient development of AI-based resource allocation algorithms.
2. Seamless integration of predictive models with cloud infrastructure.
3. Supports dynamic scaling and adaptation to changing workloads.
4. Multiple deployment options including Google Colab and AWS SageMaker.
5. Real-time monitoring and logging of resource utilization metrics.

### Directory Structure

```
├── README.md
├── baseline_arima.py
├── baseline_lstm.py
├── dataset.sh
├── deepar_modelling.py
├── feature_engineering.py
├── df_scaled.csv
└── output
    ├── CPU_cap_under_1mo.png
    ├── ac_cpu_usage.png
    ├── acf_pacf.png
    ├── adf_test.png
    ├── anomaly_detection.png
    ├── arima_act_pred.png
    ├── cpu_over_under.png
    ├── deepar_comparison_test.png
    ├── deepar_pred.png
    ├── lstm_act_pred.png
    └── seasonality_analysis.png
```

## Dataset Fetching Script

This script is designed to fetch the dataset from the Grid Workloads Archive and extract it to a specified directory.

### Prerequisites

- Ensure you have `wget` and `unzip` utilities installed on your system.
- The script should have execute permissions.

### Execution Steps

1. **Provide Execute Permissions**: Before running the script, you need to ensure it has the necessary execute permissions. You can provide these permissions using the following command:

```
chmod +x script_name.sh
```

2. **Run the Script**: Once permissions are set, you can run the script using:

```
./dataset.sh
```

## Feature Engineering and Analysis for CPU Usage

This script focuses on preparing, preprocessing, and analyzing CPU usage data. It combines data from different sources, engineers relevant features, checks for stationarity, and visualizes various aspects of the data.

### Description

The script contains various components:

- **Data Preparation**: Loads and concatenates CPU usage data from multiple CSV files.
- **Feature Engineering**: Adds several time-based features and derives new columns based on differences in CPU usage and network throughput.
- **Data Visualization**: Provides various plots such as Autocorrelation, CPU Capacity vs. Usage, ACF & PACF, and Seasonality & Trend Analysis.
- **Statistical Analysis**: Implements the Augmented Dickey-Fuller Test to check the stationarity of the series and moving average for anomaly detection.

### Prerequisites

1. **Google Colab Account**: Since this script was originally developed for Google Colab, having a Google account to access Colab is essential.
2. **Data**: Ensure you have your CSV files uploaded to Google Drive in the appropriate directory structure as indicated in the script.
3. **Python Libraries**: Make sure the following libraries are installed:
   - numpy
   - pandas
   - matplotlib
   - seaborn
   - statsmodels
   - scikit-learn

### Steps to Run in Google Colab

1. **Mount Google Drive**:

   - Open Google Colab.
   - Upload this script.
   - Run the script. It will ask for authentication to mount your Google Drive. Follow the link, authenticate, and copy the provided code. Paste it back in Colab to mount your drive.

2. **Adjust Directories**:

   - Modify the `log_dir` variable in the script to point to the directory in your Google Drive where the CSV files are located.

3. **Run the Script**:

   - Execute the entire script in Colab. As the script progresses, it will visualize various aspects of the data and conduct analyses.

4. **Save & Export**:
   - Any generated plots or saved CSV files will be stored in the directory pointed by `log_dir`. You can download or further analyze these files as needed.

### Instructions for Running Locally

Run the script:

```bash
python feature_engineering.py
```

### Note

If you wish to run the script outside of Google Colab (e.g., locally), you will need to modify the data loading sections to read from local directories rather than from Google Drive.

## CPU Workload Prediction using DeepAR on AWS SageMaker

### Description

This script is developed to predict CPU workloads using the DeepAR algorithm on AWS SageMaker. It fetches data, processes it, and trains a DeepAR model to forecast CPU usage.

### Prerequisites

- AWS Account with SageMaker and S3 access.
- IAM roles set up with permissions for SageMaker and S3.
- The dataset for the first time (optional).

### Instructions

1. **Setup SageMaker**:

   - Create a SageMaker notebook instance.
   - Once the notebook instance is active, open Jupyter or JupyterLab.
   - Upload the provided script.

2. **Install necessary libraries/packages** (if not already available in your SageMaker instance):

   - You might need to manually install some libraries if they are not pre-installed on SageMaker.

3. **Dataset Setup**:

   - If you are downloading the dataset for the first time, uncomment the lines in the script related to dataset download and extraction.

4. **Run the Script**:

   - Execute the cells of the notebook/script sequentially.

5. **Visualize the Results**:
   - The script will visualize the predicted vs actual CPU usage and will also display some evaluation metrics.

### Output

The script will:

- Process the data and visualize the first few rows.
- Train a DeepAR model on the SageMaker.
- Plot the actual vs predicted values for CPU usage.
- Display evaluation metrics including Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, and R2 Score.

## Baseline Modelling with ARIMA

This script models a CPU prediction using the ARIMA (AutoRegressive Integrated Moving Average) technique.

### Description

The script reads a dataset (`df_scaled.csv` generated by `deepar_modelling.py`) containing CPU usage data, splits it into training and testing sets, and then uses the ARIMA technique to predict CPU usage. The predictions are then compared with the actual values, and various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R2 score are computed. The actual vs. predicted CPU usage is also visualized in a plot.

### Prerequisites

- Ensure you have Python installed.
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `sklearn`, and `google-colab` (if running on Colab).
- The dataset file `df_scaled.csv` should be present in the specified Google Drive path (`/content/gdrive/My Drive/workload_predictor_vm`).

### Instructions for Running in Google Colab

1. Open Google Colab.
2. Upload the script to Google Colab.
3. Ensure the dataset file (`df_scaled.csv`) is present in your Google Drive at the specified path.
4. Execute the script in Colab.
5. The plot showing actual vs. predicted CPU usage will be saved in the specified Google Drive path.

### Instructions for Running Locally

1. Clone the repository to your local machine.
2. Navigate to the directory containing the script.
3. Ensure you have all the required libraries installed. You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn statsmodels sklearn
```

4. Execute the script:

```bash
python baseline_arima.py
```

Replace `script_name.py` with the actual name of the script.

5. The plot showing actual vs. predicted CPU usage will be saved in the specified directory path.

## Baseline Modelling with LSTM

This script models a CPU prediction using the Long Short-Term Memory (LSTM) neural network to model and predict CPU usage.

### Description

The script takes in a dataset (`df_scaled.csv` generated by `deepar_modelling.py`) which holds CPU usage details. This data undergoes normalization and is split into training and testing sets. The LSTM model is then trained on the training set and used to predict the CPU usage on the testing set. Predictions are later compared with actual values, and performance metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are computed. Finally, a plot visualizing the actual versus predicted CPU usage is generated.

### Prerequisites

- Ensure you have Python and necessary libraries installed.
- Libraries required: `numpy`, `pandas`, `matplotlib`, `seaborn`, `keras`, `sklearn`, and `google-colab` (if running on Colab).
- The dataset file `df_scaled.csv` should be available in the specified Google Drive path (`/content/gdrive/My Drive/workload_predictor_vm`).

### Instructions for Running in Google Colab

1. Open Google Colab.
2. Upload this script.
3. Confirm that the dataset file (`df_scaled.csv`) is in the specified Google Drive location.
4. Run the script within Colab.
5. Upon execution, a plot illustrating the actual vs. predicted CPU usage will be saved to the designated Google Drive directory.

### Running the Script Locally

1. Clone the repository to your machine.
2. Navigate to the directory containing the script.
3. Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn keras sklearn google-colab
```

4. Run the script:

```bash
python baseline_lstm.py
```

Make sure to replace `baseline_lstm.py` with the actual name of the script.

5. The visualization contrasting actual and predicted CPU usage will be saved in the specified directory.



### Contributors:
1. [Sasidharan V](https://github.com/sasidharan01)
2. [Subramaniam Subramanian Murugesan](https://github.com/Subramaniam-dot)
