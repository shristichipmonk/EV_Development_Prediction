import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def load_data(nm,battery): 
    mat = loadmat(nm)
    counter = 0
    dataset = []
    capacity_data = []
  
    for i in range(len(mat[battery][0, 0]['cycle'][0])):
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge' :
            ambient_temperature = row['ambient_temperature'][0][0]
            date_time = datetime.datetime(int(row['time'][0][0]),
                                   int(row['time'][0][1]),
                                   int(row['time'][0][2]),
                                   int(row['time'][0][3]),
                                   int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]
            for j in range(len(data[0][0]['Voltage_measured'][0])):
                voltage_measured = data[0][0]['Voltage_measured'][0][j]
                current_measured = data[0][0]['Current_measured'][0][j]
                temperature_measured = data[0][0]['Temperature_measured'][0][j]
                current_load = data[0][0]['Current_load'][0][j]
                voltage_load = data[0][0]['Voltage_load'][0][j]
                time = data[0][0]['Time'][0][j]
                dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                                voltage_measured, current_measured,
                                temperature_measured, current_load,
                                voltage_load, time])
                capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
                counter = counter + 1
    return [pd.DataFrame(data=dataset,
                       columns=['cycle', 'ambient_temperature', 'datetime',
                                'capacity', 'voltage_measured',
                                'current_measured', 'temperature_measured',
                                'current_load', 'voltage_load', 'time']),
          pd.DataFrame(data=capacity_data,
                       columns=['cycle', 'ambient_temperature', 'datetime',
                                'capacity'])]

B0005_dataset, B0005_capacity = load_data('B0005.mat','B0005')
B0005_dataset['flag'] = 1

B0006_dataset, B0006_capacity = load_data('B0006.mat','B0006')
B0006_dataset['flag'] = 2

B0007_dataset, B0007_capacity = load_data('B0007.mat','B0007')
B0007_dataset['flag'] = 3

B0018_dataset, B0018_capacity = load_data('B0018.mat','B0018')
B0018_dataset['flag'] = 4

fuel_cells_df = pd.concat([B0005_dataset,B0006_dataset,B0007_dataset,B0018_dataset], ignore_index = True)

fuel_cells_df.drop('ambient_temperature', axis = 1, inplace = True)

def missingValue(df):
    total_null = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/len(df))*100).sort_values(ascending = False)
    
    missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
    return missing_data
missing_df = missingValue(fuel_cells_df)

fuel_cells_df.drop_duplicates(keep = 'first', inplace = True)
fuel_cells_df_copy2 = fuel_cells_df.copy()

## 3.  Isolation Forest based Anomaly detection
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
# Feature selection (assuming these are relevant features for anomaly detection)
features = ['voltage_measured', 'current_measured', 'temperature_measured', 'time']
# Prepare the feature matrix
X = fuel_cells_df[features]
# Initialize Isolation Forest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
# Fit the model
isolation_forest.fit(X)
# Predict outliers
outliers = isolation_forest.predict(X)
# Add outlier predictions to the DataFrame
fuel_cells_df['is_outlier'] = outliers

# Visualize outliers (optional)
plt.scatter(fuel_cells_df['time'], fuel_cells_df['voltage_measured'], c=fuel_cells_df['is_outlier'], cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Voltage Measured')
plt.title('Isolation Forest Outliers')
plt.colorbar(label='Outlier')
plt.show()

# Print the outliers
print("Outlier observations:")
print(fuel_cells_df[fuel_cells_df['is_outlier'] == -1])

# Calculate average capacity retention for each battery
average_capacity_retention = fuel_cells_df.groupby('flag')['capacity'].mean()

# Calculate average cycle life for each battery
average_cycle_life = fuel_cells_df.groupby('flag')['cycle'].max()

# Print the results
print("Average Capacity Retention:")
print(average_capacity_retention)
print("\nAverage Cycle Life:")
print(average_cycle_life)

# Determine the battery with the highest average capacity retention
best_battery_capacity = average_capacity_retention.idxmax()
best_battery_cycle = average_cycle_life.idxmax()

print(f"\nThe battery with the highest average capacity retention is Battery #{best_battery_capacity}.")
print(f"The battery with the highest average cycle life is Battery #{best_battery_cycle}.")

