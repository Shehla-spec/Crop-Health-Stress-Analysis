#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opendatasets')


# In[2]:


pip install kagglehub


# In[3]:


import opendatasets as od


# In[ ]:


dataset_url = 'https://www.kaggle.com/datasets/datasetengineer/crop-health-and-environmental-stress-dataset'


# In[ ]:


od.download(dataset_url)


# In[ ]:


import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/datasetengineer/crop-health-and-environmental-stress-dataset'

# This will prompt you for your Kaggle username and API Key
od.download(dataset_url)


# In[ ]:


get_ipython().system('pip install kaggle')


# In[ ]:


import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('datasetengineer/crop-health-and-environmental-stress-dataset', path='.', unzip=True)


# In[ ]:


import sys
print(sys.version)


# In[ ]:


import os
import kaggle

# Set the token you just generated
os.environ['KAGGLE_API_TOKEN'] = "KGAT_75efefd21567aa1d5d449fb546d31395"

# Authenticate
kaggle.api.authenticate()

# Download and unzip the specific dataset
dataset_path = 'datasetengineer/crop-health-and-environmental-stress-dataset'
kaggle.api.dataset_download_files(dataset_path, path='.', unzip=True)

print("Download complete! Check your folder for the dataset files.")


# In[ ]:


import os

# 1. SET THE TOKEN FIRST
os.environ['KAGGLE_API_TOKEN'] = "KGAT_75efefd21567aa1d5d449fb546d31395"

# 2. NOW IMPORT KAGGLE
import kaggle

# 3. AUTHENTICATE AND DOWNLOAD
kaggle.api.authenticate()
kaggle.api.dataset_download_files('datasetengineer/crop-health-and-environmental-stress-dataset', path='.', unzip=True)

print("Success! Data downloaded.")


# In[ ]:


import os
import pandas as pd

# List all files in your current directory to find the CSV name
files = os.listdir('.')
print("Files in directory:", files)

# Look for the CSV file (it's usually named 'crop_health_data.csv' or similar)
csv_file = [f for f in files if f.endswith('.csv')]

if csv_file:
    print(f"Found dataset: {csv_file[0]}")
    # Load the data
    df = pd.read_csv(csv_file[0])
    print("\n--- Dataset Preview ---")
    display(df.head())
else:
    print("CSV file not found. You might need to check if the unzip was successful.")


# In[ ]:


# Check for missing values
print(df.isnull().sum())

# Statistical summary of the health indicators
display(df[['NDVI', 'Canopy_Coverage', 'Expected_Yield', 'Pest_Damage']].describe())


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

# Filtering out the negative yield outliers for a cleaner plot
clean_df = df[df['Expected_Yield'] >= 0]

plt.figure(figsize=(10, 5))
sns.scatterplot(data=clean_df.sample(1000), x='Soil_Moisture', y='NDVI', hue='Crop_Type')
plt.title("Soil Moisture vs NDVI (Sample of 1000 points)")
plt.show()


# In[ ]:


# 1. Remove negative yields
df = df[df['Expected_Yield'] >= 0]

# 2. Clip NDVI to its physical maximum of 1.0
df['NDVI'] = df['NDVI'].clip(upper=1.0)

print(f"Data cleaned. Remaining rows: {len(df)}")


# In[ ]:


# Correlation of all numeric features with Crop Health Label
# Note: This assumes Crop_Health_Label is numeric (0/1)
health_corr = df.select_dtypes(include=['number']).corr()['Crop_Health_Label'].sort_values(ascending=False)
print("Top factors influencing Crop Health:")
print(health_corr)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Encode the 'Crop_Type' (Wheat, Maize, Rice) into numbers
le = LabelEncoder()
df['Crop_Type_Encoded'] = le.fit_transform(df['Crop_Type'])

# 2. Select your Features (X) and your Target (y)
features = [
    'NDVI', 'SAVI', 'Chlorophyll_Content', 'Soil_Moisture', 
    'Temperature', 'Rainfall', 'Pest_Damage', 'Crop_Type_Encoded', 
    'Canopy_Coverage', 'Soil_pH'
]

X = df[features]
y = df['Expected_Yield']

# 3. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples...")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize the model (using 100 trees)
# n_jobs=-1 uses all your CPU cores to speed it up
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

print("Step 1: Preparing Features...")
# Selecting features and target
features = ['NDVI', 'SAVI', 'Chlorophyll_Content', 'Soil_Moisture', 
            'Temperature', 'Rainfall', 'Pest_Damage', 'Canopy_Coverage', 'Soil_pH']

X = df[features]
y = df['Expected_Yield']

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Step 2: Split complete. Training on {len(X_train)} rows...")

# Step 3: Initialize and Train (Using 20 trees for a faster test)
model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)

print("Step 3: Training Model (This may take 30-60 seconds)...")
model.fit(X_train, y_train)

# Step 4: Predict and Score
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Results ---")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

print("Step 1: Preparing Features...")
# Selecting features and target
features = ['NDVI', 'SAVI', 'Chlorophyll_Content', 'Soil_Moisture', 
            'Temperature', 'Rainfall', 'Pest_Damage', 'Canopy_Coverage', 'Soil_pH']

X = df[features]
y = df['Expected_Yield']

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Step 2: Split complete. Training on {len(X_train)} rows...")

# Step 3: Initialize and Train (Using 20 trees for a faster test)
model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)

print("Step 3: Training Model (This may take 30-60 seconds)...")
model.fit(X_train, y_train)

# Step 4: Predict and Score
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Results ---")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

print("Step 1: Preparing Features...")
# Selecting features and target
features = ['NDVI', 'SAVI', 'Chlorophyll_Content', 'Soil_Moisture', 
            'Temperature', 'Rainfall', 'Pest_Damage', 'Canopy_Coverage', 'Soil_pH']

X = df[features]
y = df['Expected_Yield']

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Step 2: Split complete. Training on {len(X_train)} rows...")

# Step 3: Initialize and Train (Using 20 trees for a faster test)
model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)

print("Step 3: Training Model (This may take 30-60 seconds)...")
model.fit(X_train, y_train)

# Step 4: Predict and Score
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Results ---")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")


# In[ ]:


import pandas as pd

# Load the file again (using the name we found in your directory earlier)
df = pd.read_csv('agriculture_dataset.csv')

# Clean the impossible values (Essential for the Biology Expert role!)
df = df[df['Expected_Yield'] >= 0]
df['NDVI'] = df['NDVI'].clip(upper=1.0)

print(f"Data re-loaded and cleaned. Total rows: {len(df)}")


# In[ ]:


import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Scientific Correlation: Does Chlorophyll match Leaf Health (NDVI)?
correlation, p_value = stats.pearsonr(df['NDVI'], df['Chlorophyll_Content'])

print(f"Pearson Correlation: {correlation:.4f}")
print(f"P-Value: {p_value:.4e}")

# Visualizing the Biological Relationship
plt.figure(figsize=(8, 5))
sns.regplot(data=df.sample(2000), x='Chlorophyll_Content', y='NDVI', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("Numerical Validation: Chlorophyll vs. NDVI")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Selecting the 'Biological Drivers'
bio_features = ['NDVI', 'Chlorophyll_Content', 'Soil_Moisture', 'Temperature', 'Rainfall', 'Soil_pH']
X = df[bio_features]
y = df['Expected_Yield']

# Scaling (Essential for "Numerical Validation" experience)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Model (Using 50 trees for a balance of speed and accuracy)
yield_model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
yield_model.fit(X_train, y_train)

print("Model Training Complete!")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Selecting the 'Biological Drivers'
bio_features = ['NDVI', 'Chlorophyll_Content', 'Soil_Moisture', 'Temperature', 'Rainfall', 'Soil_pH']
X = df[bio_features]
y = df['Expected_Yield']

# Scaling (Essential for "Numerical Validation" experience)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Model (Using 50 trees for a balance of speed and accuracy)
yield_model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
yield_model.fit(X_train, y_train)

print("Model Training Complete!")


# In[ ]:


import pandas as pd

# Load the file again (using the name we found in your directory earlier)
df = pd.read_csv('agriculture_dataset.csv')

# Clean the impossible values (Essential for the Biology Expert role!)
df = df[df['Expected_Yield'] >= 0]
df['NDVI'] = df['NDVI'].clip(upper=1.0)

print(f"Data re-loaded and cleaned. Total rows: {len(df)}")


# In[ ]:


import pandas as pd

# Load the file again (using the name we found in your directory earlier)
df = pd.read_csv('agriculture_dataset.csv')

# Clean the impossible values (Essential for the Biology Expert role!)
df = df[df['Expected_Yield'] >= 0]
df['NDVI'] = df['NDVI'].clip(upper=1.0)

print(f"Data re-loaded and cleaned. Total rows: {len(df)}")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Selecting the 'Biological Drivers'
bio_features = ['NDVI', 'Chlorophyll_Content', 'Soil_Moisture', 'Temperature', 'Rainfall', 'Soil_pH']
X = df[bio_features]
y = df['Expected_Yield']

# Scaling (Essential for "Numerical Validation" experience)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Model (Using 50 trees for a balance of speed and accuracy)
yield_model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
yield_model.fit(X_train, y_train)

print("Model Training Complete!")


# In[ ]:


# Extracting and visualizing importance
importance = pd.Series(yield_model.feature_importances_, index=bio_features)
importance.sort_values().plot(kind='barh', color='teal')
plt.title("Which Biological Factors Predict Yield?")
plt.xlabel("Relative Importance Score")
plt.show()


# In[6]:


import pandas as pd

# Load the file again (using the name we found in your directory earlier)
df = pd.read_csv('agriculture_dataset.csv')

# Clean the impossible values (Essential for the Biology Expert role!)
df = df[df['Expected_Yield'] >= 0]
df['NDVI'] = df['NDVI'].clip(upper=1.0)

print(f"Data re-loaded and cleaned. Total rows: {len(df)}")


# In[7]:


# Extracting and visualizing importance
importance = pd.Series(yield_model.feature_importances_, index=bio_features)
importance.sort_values().plot(kind='barh', color='teal')
plt.title("Which Biological Factors Predict Yield?")
plt.xlabel("Relative Importance Score")
plt.show()


# In[8]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. LOAD & CLEAN (Numerical Validation)
df = pd.read_csv('agriculture_dataset.csv')
df = df[df['Expected_Yield'] >= 0] # Remove biological impossibilities
df['NDVI'] = df['NDVI'].clip(upper=1.0) 

# 2. FEATURE SELECTION
bio_features = ['NDVI', 'Chlorophyll_Content', 'Soil_Moisture', 'Temperature', 'Rainfall', 'Soil_pH']
X = df[bio_features]
y = df['Expected_Yield']

# 3. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL TRAINING (Reduced trees to 20 for speed)
print("Training the Bio-Predictive Model...")
yield_model = RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=42)
yield_model.fit(X_train, y_train)

# 5. VISUALIZATION (The "Result" for your Portfolio)
print("Generating Importance Plot...")
importance = pd.Series(yield_model.feature_importances_, index=bio_features)
importance.sort_values().plot(kind='barh', color='teal')
plt.title("Which Biological Factors Predict Yield?")
plt.xlabel("Relative Importance Score")
plt.show()

print("Process Complete!")


# In[9]:


plt.savefig('my_bio_analysis.png')


# In[10]:


import matplotlib.pyplot as plt
import pandas as pd

# 1. Create the figure explicitly
plt.figure(figsize=(10, 6))

# 2. Sort and Plot the data 
# (Make sure 'importance' and 'bio_features' were defined in your model step)
plot_data = pd.Series(yield_model.feature_importances_, index=bio_features).sort_values()
plot_data.plot(kind='barh', color='teal')

# 3. Add the "Scientific" labels
plt.title("Numerical Validation: Biological Drivers of Expected Yield", fontsize=14)
plt.xlabel("Gini Importance (Relative Contribution)", fontsize=12)
plt.ylabel("Environmental Factors", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 4. SAVE IT FIRST (This prevents the "0 Axes" blank file)
plt.savefig('Yield_Importance_Final.png', dpi=300, bbox_inches='tight')

# 5. SHOW IT SECOND
plt.show()

print("Check your folder for 'Yield_Importance_Final.png'!")


# In[12]:


import os
# This command tells Windows to open the current folder in File Explorer
os.startfile(os.getcwd())


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# --- 1. BIOLOGICAL VALIDATION (NDVI vs Chlorophyll) ---
plt.figure(figsize=(8, 5))
sns.regplot(data=df.sample(2000), x='Chlorophyll_Content', y='NDVI', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("Validation: Chlorophyll vs. NDVI Correlation")
plt.savefig('1_Biological_Validation.png', dpi=300, bbox_inches='tight')
plt.show() # Clears the board for the next one

# --- 2. ENVIRONMENTAL INTERACTION (Heatmap) ---
plt.figure(figsize=(10, 8))
# Only use numeric columns for the heatmap
numeric_df = df[['NDVI', 'Chlorophyll_Content', 'Soil_Moisture', 'Temperature', 'Rainfall', 'Expected_Yield']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdYlGn', center=0)
plt.title("Environmental Interaction Heatmap")
plt.savefig('2_Environmental_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show() # Clears the board again

# --- 3. PREDICTIVE DRIVERS (Feature Importance) ---
plt.figure(figsize=(10, 6))
importance = pd.Series(yield_model.feature_importances_, index=bio_features).sort_values()
importance.plot(kind='barh', color='teal')
plt.title("Yield Drivers: Machine Learning Analysis")
plt.savefig('3_Yield_Importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Success! Check this folder: {os.getcwd()}")


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prepare data
clean_df = df[df['Expected_Yield'] >= 0]

# 2. Create the figure
plt.figure(figsize=(10, 5))

# 3. Create the scatter plot
sns.scatterplot(data=clean_df.sample(1000), x='Soil_Moisture', y='NDVI', hue='Crop_Type')

# 4. Add professional formatting
plt.title("Biophysical Relationship: Soil Moisture vs NDVI", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# 5. SAVE THE FILE (Before plt.show())
plt.savefig('4_Soil_vs_NDVI.png', dpi=300, bbox_inches='tight')

# 6. Display it in Jupyter
plt.show()

print("File '4_Soil_vs_NDVI.png' has been saved to your project folder.")


# In[ ]:




