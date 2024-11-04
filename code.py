import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data=pd.read_csv('/content/blood_count_dataset.csv')
data.head()
data.info()
from google.colab import drive
drive.mount('/content/drive')
x=data.iloc[:,:1].values
y=data.iloc[:,:1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#print(y_pred)
#print(y_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,model.predict(x_train),color='blue')
plt.title('Blood Count Prediction')
#plt.xlabel('')
#plt.ylabel('')
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the file is in your Google Drive, you need to mount your Drive first:
from google.colab import drive
drive.mount('/content/drive')

# Update the path to include your Drive's folder structure
# Replace 'My Drive/Data' with the actual path to your file within your Drive
file_path = '/content/drive/My Drive/Data/blood_count_dataset.csv'

df = pd.read_csv('/content/blood_count_dataset.csv')
sns.pairplot(df)
plt.show()
!pip install scikeras
df.info()
x=df.iloc[:,1].values
y=df.iloc[:,0].values
x=pd.get_dummies(x,drop_first=True)
x.head()
x_train,x_tes,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("x_train:",x_train.shape)
print("x_test:",x_test.shape)
print("y_train:",y_train.shape)
print("y_test:",y_test.shape)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor

df = pd.read_csv('/content/blood_count_dataset.csv')
df.head()

features = ['Age', 'Gender']
# Corrected the column names in the 'targets' list
targets = ['Hemoglobin', 'Platelets', 'WBC', 'RBC', 'MCV', 'MCH', 'MCHC']

x = df[features]
y = df[targets]
x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Convert x_train and x_test to int64
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)

def build_model():
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(len(targets), kernel_initializer='uniform'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor

# Define the model-building function
def build_model():
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(len(targets), kernel_initializer='uniform'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create KerasRegressor
model = KerasRegressor(model=build_model, batch_size=15, epochs=100)

# Train the model
model.fit(x_train, y_train)

# Save the underlying Keras model in H5 format
model.model_.save('C:/Users/USER/Downloads/patientprediction/blood_count_data_model.h5')

# Make predictions
y_pred = model.predict(x_test)

# Optionally, you can print or log the predictions
print("Predictions:", y_pred)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

age = float(input("Enter your age: "))
gender = input("Enter your gender (male/female): ")
new_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender]
})
new_data = pd.get_dummies(new_data, drop_first=True)
missing_cols = set(x_train.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[x_train.columns]

# Convert 'Age' column in new_data to int64 to match the model's input type
new_data['Age'] = new_data['Age'].astype(np.int64)

prediction = model.predict(new_data)
print("Predicted Hemoglobin:", prediction[0][0])
print("Predicted Platelet Count:", prediction[0][1])
print("Predicted White Blood Cells:", prediction[0][2])
print("Predicted Red Blood Cells:", prediction[0][3])
print("Predicted MCV:", prediction[0][4])
print("Predicted MCH:", prediction[0][5])
print("Predicted MCHC:", prediction[0][6])
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
import pickle  # Import pickle for saving other objects

# Load your dataset
df = pd.read_csv('/content/blood_count_dataset.csv')

# Define features and targets
features = ['Age', 'Gender']
targets = ['Hemoglobin', 'Platelets', 'WBC', 'RBC', 'MCV', 'MCH', 'MCHC']

x = df[features]
y = df[targets]
x = pd.get_dummies(x, drop_first=True)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert x_train and x_test to int64
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)

# Build the model
def build_model():
    model = Sequential()
    model.add(Dense(64, kernel_initializer='uniform', activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(rate=0.2))
    model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(len(targets), kernel_initializer='uniform'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create KerasRegressor
model = KerasRegressor(model=build_model, batch_size=15, epochs=100)

# Train the model
model.fit(x_train, y_train)

# Save the underlying Keras model in H5 format
model.model_.save('C:/Users/USER/Downloads/patientprediction/blood_count_data_model.h5')

# Save training data and model details using pickle
with open('C:/Users/USER/Downloads/patientprediction/training_data.pkl', 'wb') as f:
    pickle.dump((x_train, y_train), f)

# Make predictions
y_pred = model.predict(x_test)

# Print predictions
print("Predictions:", y_pred)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# User input for prediction
age = float(input("Enter your age: "))
gender = input("Enter your gender (male/female): ")
new_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender]
})
new_data = pd.get_dummies(new_data, drop_first=True)
missing_cols = set(x_train.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[x_train.columns]

# Convert 'Age' column in new_data to int64 to match the model's input type
new_data['Age'] = new_data['Age'].astype(np.int64)

# Make predictions on new data
prediction = model.predict(new_data)
print("Predicted Hemoglobin:", prediction[0][0])
print("Predicted Platelet Count:", prediction[0][1])
print("Predicted White Blood Cells:", prediction[0][2])
print("Predicted Red Blood Cells:", prediction[0][3])
print("Predicted MCV:", prediction[0][4])
print("Predicted MCH:", prediction[0][5])
print("Predicted MCHC:", prediction[0][6])
def analyze_health(predicted_values):
    # Extract predicted values
    hemoglobin = predicted_values['hemoglobin']
    platelet_count = predicted_values['platelet_count']
    white_blood_cells = predicted_values['white_blood_cells']
    red_blood_cells = predicted_values['red_blood_cells']
    mcv = predicted_values['mcv']
    mch = predicted_values['mch']
    mchc = predicted_values['mchc']

    # Initialize health status and severity
    health_status = "Healthy"
    severity = 0
    suggestions = []

    # Analyze each parameter based on common medical standards
    if hemoglobin < 12 or hemoglobin > 16:  # Example range for females
        health_status = "Unhealthy"
        severity += 2  # Increase severity
        suggestions.append("Consult a doctor regarding hemoglobin levels.")

    if platelet_count < 150000 or platelet_count > 450000:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Monitor your platelet count and consult a healthcare provider.")

    if white_blood_cells < 4000 or white_blood_cells > 10000:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consider lifestyle changes to boost immune health.")

    if red_blood_cells < 4.0 or red_blood_cells > 5.5:  # Example range for females
        health_status = "Unhealthy"
        severity += 2
        suggestions.append("Discuss your RBC levels with a healthcare professional.")

    if mcv < 80 or mcv > 100:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consider dietary changes or supplements.")

    if mch < 27 or mch > 32:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consult a healthcare provider regarding MCH levels.")

    if mchc < 32 or mchc > 36:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Monitor MCHC levels and consider dietary adjustments.")

    # Map severity to a scale of 1 to 5
    severity = min(severity, 5)

    # Output results
    return {
        'health_status': health_status,
        'severity': severity,
        'suggestions': suggestions
    }

# Example predicted values
predicted_values = {
    'hemoglobin': 32.49608,
    'platelet_count': 429036.2,
    'white_blood_cells': 10337.558,
    'red_blood_cells': -9.799953,
    'mcv': 182.14523,
    'mch': 38.30445,
    'mchc': 42.964577
}

# Analyze the health status
health_analysis = analyze_health(predicted_values)

# Display results
print(f"Health Status: {health_analysis['health_status']}")
print(f"Severity (out of 5): {health_analysis['severity']}")
print("Suggestions for Improvement:")
for suggestion in health_analysis['suggestions']:
    print(f"- {suggestion}")
def analyze_health(predicted_values):
    # Extract predicted values
    hemoglobin = predicted_values['hemoglobin']
    platelet_count = predicted_values['platelet_count']
    white_blood_cells = predicted_values['white_blood_cells']
    red_blood_cells = predicted_values['red_blood_cells']
    mcv = predicted_values['mcv']
    mch = predicted_values['mch']
    mchc = predicted_values['mchc']

    # Initialize health status and severity
    health_status = "Healthy"
    severity = 0
    suggestions = []

    # Analyze each parameter based on common medical standards
    if hemoglobin < 12 or hemoglobin > 16:  # Example range for females
        health_status = "Unhealthy"
        severity += 2
        suggestions.append("Consult a doctor regarding hemoglobin levels.")

    if platelet_count < 150000 or platelet_count > 450000:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Monitor your platelet count and consult a healthcare provider.")

    if white_blood_cells < 4000 or white_blood_cells > 10000:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consider lifestyle changes to boost immune health.")

    if red_blood_cells < 4.0 or red_blood_cells > 5.5:  # Example range for females
        health_status = "Unhealthy"
        severity += 2
        suggestions.append("Discuss your RBC levels with a healthcare professional.")

    if mcv < 80 or mcv > 100:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consider dietary changes or supplements.")

    if mch < 27 or mch > 32:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consult a healthcare provider regarding MCH levels.")

    if mchc < 32 or mchc > 36:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Monitor MCHC levels and consider dietary adjustments.")

    # Map severity to a scale of 1 to 5
    severity = min(severity, 5)

    # Output results
    return {
        'health_status': health_status,
        'severity': severity,
        'suggestions': suggestions
    }

# Get user input for predicted values
predicted_values = {
    'hemoglobin': float(input("Enter predicted Hemoglobin: ")),
    'platelet_count': float(input("Enter predicted Platelet Count: ")),
    'white_blood_cells': float(input("Enter predicted White Blood Cells: ")),
    'red_blood_cells': float(input("Enter predicted Red Blood Cells: ")),
    'mcv': float(input("Enter predicted MCV: ")),
    'mch': float(input("Enter predicted MCH: ")),
    'mchc': float(input("Enter predicted MCHC: "))
}

# Analyze the health status
health_analysis = analyze_health(predicted_values)

# Display results
print(f"\nHealth Status: {health_analysis['health_status']}")
print(f"Severity (out of 5): {health_analysis['severity']}")
print("Suggestions for Improvement:")
for suggestion in health_analysis['suggestions']:
    print(f"- {suggestion}")
def analyze_health(predicted_values):
    # Extract predicted values
    hemoglobin = predicted_values['hemoglobin']
    platelet_count = predicted_values['platelet_count']
    white_blood_cells = predicted_values['white_blood_cells']
    red_blood_cells = predicted_values['red_blood_cells']
    mcv = predicted_values['mcv']
    mch = predicted_values['mch']
    mchc = predicted_values['mchc']

    # Initialize health status and severity
    health_status = "Healthy"
    severity = 0
    suggestions = []

    # Analyze each parameter based on common medical standards
    if hemoglobin < 12 or hemoglobin > 16:  # Example range for females
        health_status = "Unhealthy"
        severity += 2
        suggestions.append("Consult a doctor regarding hemoglobin levels.")

    if platelet_count < 150000 or platelet_count > 450000:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Monitor your platelet count and consult a healthcare provider.")

    if white_blood_cells < 4000 or white_blood_cells > 10000:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consider lifestyle changes to boost immune health.")

    if red_blood_cells < 4.0 or red_blood_cells > 5.5:  # Example range for females
        health_status = "Unhealthy"
        severity += 2
        suggestions.append("Discuss your RBC levels with a healthcare professional.")

    if mcv < 80 or mcv > 100:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consider dietary changes or supplements.")

    if mch < 27 or mch > 32:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Consult a healthcare provider regarding MCH levels.")

    if mchc < 32 or mchc > 36:  # Normal range
        health_status = "Unhealthy"
        severity += 1
        suggestions.append("Monitor MCHC levels and consider dietary adjustments.")

    # Map severity to a scale of 1 to 5
    severity = min(severity, 5)

    # Output results
    return {
        'health_status': health_status,
        'severity': severity,
        'suggestions': suggestions
    }

# Predicted values from previous output
predicted_values = {
    'hemoglobin': 32.49608,
    'platelet_count': 429036.2,
    'white_blood_cells': 10337.558,
    'red_blood_cells': -9.799953,
    'mcv': 182.14523,
    'mch': 38.30445,
    'mchc': 42.964577
}

# Analyze the health status
health_analysis = analyze_health(predicted_values)

# Display results
print(f"\nHealth Status: {health_analysis['health_status']}")
print(f"Severity (out of 5): {health_analysis['severity']}")
print("Suggestions for Improvement:")
for suggestion in health_analysis['suggestions']:
    print(f"- {suggestion}")

import matplotlib.pyplot as plt

# Health ranges (for a healthy individual)
health_ranges = {
    'hemoglobin': (12, 16),
    'platelet_count': (150000, 450000),
    'white_blood_cells': (4000, 10000),
    'red_blood_cells': (4.0, 5.5),
    'mcv': (80, 100),
    'mch': (27, 32),
    'mchc': (32, 36)
}

# Predicted values from previous output
predicted_values = {
    'hemoglobin': 32.49608,
    'platelet_count': 429036.2,
    'white_blood_cells': 10337.558,
    'red_blood_cells': -9.799953,
    'mcv': 182.14523,
    'mch': 38.30445,
    'mchc': 42.964577
}

# Prepare data for plotting
parameters = list(health_ranges.keys())
healthy_min = [health_ranges[param][0] for param in parameters]
healthy_max = [health_ranges[param][1] for param in parameters]
predicted = [predicted_values[param] for param in parameters]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(parameters))

# Healthy range (shaded area)
ax.fill_between(x, healthy_min, healthy_max, color="lightgreen", alpha=0.3, label="Healthy Range")

# Predicted values plot
ax.plot(x, predicted, color="red", marker='o', label="Predicted Values")

# Customizing plot
ax.set_xticks(x)
ax.set_xticklabels(parameters, rotation=45)
ax.set_ylabel("Values")
ax.set_title("Comparison of Predicted Values with Healthy Ranges")
ax.legend()

plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Define the normal reference ranges and predicted values (as per the provided code)
parameters = ['Hemoglobin', 'Platelets', 'WBC', 'RBC', 'MCV', 'MCH', 'MCHC']
normal_ranges_min = [12, 150000, 4000, 4., 80, 27, 32]  # Lower bound of normal ranges
normal_ranges_max = [16, 450000, 10000, 5.5, 100, 32, 36] # Upper bound of normal ranges
predicted_values = [32.49608, 429036.2, 10337.558, -9.799953, 182.14523, 38.30445, 42.964577]

# Plot for Healthy Reference Ranges
plt.figure(figsize=(10, 6))
x = np.arange(len(parameters))
plt.bar(x - 0.2, normal_ranges_min, width=0.4, label='Normal Min', color='skyblue')
plt.bar(x + 0.2, normal_ranges_max, width=0.4, label='Normal Max', color='dodgerblue')
plt.xlabel("Health Parameters")
plt.ylabel("Values")
plt.title("Healthy Person's Blood Health Parameter Ranges")
plt.xticks(x, parameters)
plt.legend()
plt.show()

# Plot for Predicted Values
plt.figure(figsize=(10, 6))
plt.bar(parameters, predicted_values, color='salmon')
plt.xlabel("Health Parameters")
plt.ylabel("Values")
plt.title("Predicted Person's Blood Health Parameters")
plt.xticks(rotation=45)
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Define the normal reference ranges and predicted values (as per the provided code)
parameters = ['Hemoglobin', 'Platelet Count', 'White Blood Cells', 'Red Blood Cells', 'MCV', 'MCH', 'MCHC']
normal_ranges_min = [12, 150000, 4000, 4.0, 80, 27, 32]  # Lower bound of normal ranges
normal_ranges_max = [16, 450000, 10000, 5.5, 100, 32, 36] # Upper bound of normal ranges
predicted_values = [32.49608, 429036.2, 10337.558, -9.799953, 182.14523, 38.30445, 42.964577]

# Plot for Healthy Reference Ranges
plt.figure(figsize=(10, 6))
x = np.arange(len(parameters))
plt.bar(x - 0.2, normal_ranges_min, width=0.4, label='Normal Min', color='skyblue')
plt.bar(x + 0.2, normal_ranges_max, width=0.4, label='Normal Max', color='dodgerblue')
plt.xlabel("Health Parameters")
plt.ylabel("Values")
plt.title("Healthy Person's Blood Health Parameter Ranges")
plt.xticks(x, parameters)
plt.legend()
plt.show()

# Plot for Predicted Values
plt.figure(figsize=(10, 6))
plt.bar(parameters, predicted_values, color='salmon')
plt.xlabel("Health Parameters")
plt.ylabel("Values")
plt.title("Predicted Person's Blood Health Parameters")
plt.xticks(rotation=45)
plt.show()
