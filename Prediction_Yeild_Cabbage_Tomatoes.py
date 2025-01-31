import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Încărcarea datelor din CSV
data = pd.read_csv('path_to_your_data.csv') 

# Filtrarea datelor pentru varză și roșii
data_cabbages = data[data['Item'] == 'Cabbages']
data_tomatoes = data[data['Item'] == 'Tomatoes']

# Preprocesarea datelor pentru varză
X_cabbages = data_cabbages[['Year']]
y_area_cabbages = data_cabbages['Area Harvested (ha)']
y_prod_cabbages = data_cabbages['Production (t)'] / 100000
y_yield_cabbages = data_cabbages['Yield (kg/ha)']

# Preprocesarea datelor pentru roșii
X_tomatoes = data_tomatoes[['Year']]
y_area_tomatoes = data_tomatoes['Area Harvested (ha)']
y_prod_tomatoes = data_tomatoes['Production (t)'] / 100000
y_yield_tomatoes = data_tomatoes['Yield (kg/ha)']

# Împărțirea datelor în seturi de antrenament și test pentru varză
X_train_cabbages, X_test_cabbages, y_area_train_cabbages, y_area_test_cabbages = train_test_split(X_cabbages, y_area_cabbages, test_size=0.2, random_state=42)
X_train_cabbages, X_test_cabbages, y_prod_train_cabbages, y_prod_test_cabbages = train_test_split(X_cabbages, y_prod_cabbages, test_size=0.2, random_state=42)
X_train_cabbages, X_test_cabbages, y_yield_train_cabbages, y_yield_test_cabbages = train_test_split(X_cabbages, y_yield_cabbages, test_size=0.2, random_state=42)

# Împărțirea datelor în seturi de antrenament și test pentru roșii
X_train_tomatoes, X_test_tomatoes, y_area_train_tomatoes, y_area_test_tomatoes = train_test_split(X_tomatoes, y_area_tomatoes, test_size=0.2, random_state=42)
X_train_tomatoes, X_test_tomatoes, y_prod_train_tomatoes, y_prod_test_tomatoes = train_test_split(X_tomatoes, y_prod_tomatoes, test_size=0.2, random_state=42)
X_train_tomatoes, X_test_tomatoes, y_yield_train_tomatoes, y_yield_test_tomatoes = train_test_split(X_tomatoes, y_yield_tomatoes, test_size=0.2, random_state=42)

# Normalizarea datelor pentru îmbunătățirea performanței modelului
scaler_cabbages = StandardScaler()
X_train_cabbages_scaled = scaler_cabbages.fit_transform(X_train_cabbages)
X_test_cabbages_scaled = scaler_cabbages.transform(X_test_cabbages)

scaler_tomatoes = StandardScaler()
X_train_tomatoes_scaled = scaler_tomatoes.fit_transform(X_train_tomatoes)
X_test_tomatoes_scaled = scaler_tomatoes.transform(X_test_tomatoes)

# Crearea și antrenarea modelelor de regresie liniară pentru varză
model_area_cabbages = LinearRegression()
model_prod_cabbages = LinearRegression()
model_yield_cabbages = LinearRegression()

model_area_cabbages.fit(X_train_cabbages_scaled, y_area_train_cabbages)
model_prod_cabbages.fit(X_train_cabbages_scaled, y_prod_train_cabbages)
model_yield_cabbages.fit(X_train_cabbages_scaled, y_yield_train_cabbages)

# Crearea și antrenarea modelelor de regresie liniară pentru roșii
model_area_tomatoes = LinearRegression()
model_prod_tomatoes = LinearRegression()
model_yield_tomatoes = LinearRegression()

model_area_tomatoes.fit(X_train_tomatoes_scaled, y_area_train_tomatoes)
model_prod_tomatoes.fit(X_train_tomatoes_scaled, y_prod_train_tomatoes)
model_yield_tomatoes.fit(X_train_tomatoes_scaled, y_yield_train_tomatoes)

# Predicții pe seturile de test pentru varză și roșii
y_area_pred_cabbages = model_area_cabbages.predict(X_test_cabbages_scaled)
y_prod_pred_cabbages = model_prod_cabbages.predict(X_test_cabbages_scaled)
y_yield_pred_cabbages = model_yield_cabbages.predict(X_test_cabbages_scaled)

y_area_pred_tomatoes = model_area_tomatoes.predict(X_test_tomatoes_scaled)
y_prod_pred_tomatoes = model_prod_tomatoes.predict(X_test_tomatoes_scaled)
y_yield_pred_tomatoes = model_yield_tomatoes.predict(X_test_tomatoes_scaled)

# Creare interval de predicție pentru anii 2025-2035
years_future = np.arange(2025, 2036).reshape(-1, 1)

# Scalarea anilor pentru predicție
years_future_scaled_cabbages = scaler_cabbages.transform(years_future)
years_future_scaled_tomatoes = scaler_tomatoes.transform(years_future)

# Predicții pentru anii 2025-2035 pentru varză și roșii
y_area_pred_future_cabbages = model_area_cabbages.predict(years_future_scaled_cabbages)
y_prod_pred_future_cabbages = model_prod_cabbages.predict(years_future_scaled_cabbages)
y_yield_pred_future_cabbages = model_yield_cabbages.predict(years_future_scaled_cabbages)

y_area_pred_future_tomatoes = model_area_tomatoes.predict(years_future_scaled_tomatoes)
y_prod_pred_future_tomatoes = model_prod_tomatoes.predict(years_future_scaled_tomatoes)
y_yield_pred_future_tomatoes = model_yield_tomatoes.predict(years_future_scaled_tomatoes)

# Vizualizări grafice pentru varză
plt.figure(figsize=(18, 12))

# Grafic linie pentru Area Harvested (varză)
plt.subplot(2, 3, 1)
plt.plot(data_cabbages['Year'], y_area_cabbages, label='Real Area Harvested', color='blue')
plt.title('Cabbages - Area Harvested (Real)')
plt.xlabel('Year')
plt.ylabel('Area Harvested (ha)')
plt.legend()

# Grafic linie pentru Production (varză)
plt.subplot(2, 3, 2)
plt.plot(data_cabbages['Year'], y_prod_cabbages, label='Real Production', color='blue')
plt.title('Cabbages - Production (Real)')
plt.xlabel('Year')
plt.ylabel('Production - Hundreds of thousands of tons')
plt.legend()

# Grafic linie pentru Yield (varză)
plt.subplot(2, 3, 3)
plt.plot(data_cabbages['Year'], y_yield_cabbages, label='Real Yield', color='blue')
plt.title('Cabbages - Yield (Real)')
plt.xlabel('Year')
plt.ylabel('Yield (kg/ha)')
plt.legend()

# Grafic linie pentru predicția Area Harvested (varză)
plt.subplot(2, 3, 4)
plt.plot(years_future, y_area_pred_future_cabbages, label='Predicted Area Harvested (2025-2035)', color='red')
plt.title('Cabbages - Area Harvested (Predicted 2025-2035)')
plt.xlabel('Year')
plt.ylabel('Area Harvested (ha)')
plt.ylim(min(y_area_pred_future_cabbages)*0.9, max(y_area_pred_future_cabbages)*1.1)
for i in range(0, len(years_future), 2):  # Pas de 2 pentru a afisa din 2 in 2 ani
    plt.text(years_future[i], y_area_pred_future_cabbages[i], 
             f'{y_area_pred_future_cabbages[i]:.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')
plt.legend()

# Grafic linie pentru predicția Production (varză)
plt.subplot(2, 3, 5)
plt.plot(years_future, y_prod_pred_future_cabbages, label='Predicted Production (2025-2035)', color='red')
plt.title('Cabbages - Production (Predicted 2025-2035)')
plt.xlabel('Year')
plt.ylabel('Production - Hundreds of thousands of tons')
plt.ylim(min(y_prod_pred_future_cabbages)*0.9, max(y_prod_pred_future_cabbages)*1.1)
for i in range(0, len(years_future), 2):  # Pas de 2 pentru a afisa din 2 in 2 ani
    plt.text(years_future[i], y_prod_pred_future_cabbages[i], 
             f'{y_prod_pred_future_cabbages[i]:.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')
plt.legend()

# Grafic linie pentru predicția Yield (varză)
plt.subplot(2, 3, 6)
plt.plot(years_future, y_yield_pred_future_cabbages, label='Predicted Yield (2025-2035)', color='red')
plt.title('Cabbages - Yield (Predicted 2025-2035)')
plt.xlabel('Year')
plt.ylabel('Yield (kg/ha)')
plt.ylim(min(y_yield_pred_future_cabbages)*0.9, max(y_yield_pred_future_cabbages)*1.1)
for i in range(0, len(years_future), 2):  # Pas de 2 pentru a afisa din 2 in 2 ani
    plt.text(years_future[i], y_yield_pred_future_cabbages[i], 
             f'{y_yield_pred_future_cabbages[i]:.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')
plt.legend()

plt.tight_layout()
plt.show()

# Vizualizări grafice pentru roșii
plt.figure(figsize=(18, 12))

# Grafic linie pentru Area Harvested (roșii)
plt.subplot(2, 3, 1)
plt.plot(data_tomatoes['Year'], y_area_tomatoes, label='Real Area Harvested', color='blue')
plt.title('Tomatoes - Area Harvested (Real)')
plt.xlabel('Year')
plt.ylabel('Area Harvested (ha)')
plt.legend()

# Grafic linie pentru Production (roșii)
plt.subplot(2, 3, 2)
plt.plot(data_tomatoes['Year'], y_prod_tomatoes, label='Real Production', color='blue')
plt.title('Tomatoes - Production (Real)')
plt.xlabel('Year')
plt.ylabel('Production - Hundreds of thousands of tons')
plt.legend()

# Grafic linie pentru Yield (roșii)
plt.subplot(2, 3, 3)
plt.plot(data_tomatoes['Year'], y_yield_tomatoes, label='Real Yield', color='blue')
plt.title('Tomatoes - Yield (Real)')
plt.xlabel('Year')
plt.ylabel('Yield (kg/ha)')
plt.legend()

# Grafic linie pentru predicția Area Harvested (roșii)
plt.subplot(2, 3, 4)
plt.plot(years_future, y_area_pred_future_tomatoes, label='Predicted Area Harvested (2025-2035)', color='red')
plt.title('Tomatoes - Area Harvested (Predicted 2025-2035)')
plt.xlabel('Year')
plt.ylabel('Area Harvested (ha)')
plt.ylim(min(y_area_pred_future_tomatoes)*0.9, max(y_area_pred_future_tomatoes)*1.1)
for i in range(0, len(years_future), 2):  # Pas de 2 pentru a afisa din 2 in 2 ani
    plt.text(years_future[i], y_area_pred_future_tomatoes[i], 
             f'{y_area_pred_future_tomatoes[i]:.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')
plt.legend()

# Grafic linie pentru predicția Production (roșii)
plt.subplot(2, 3, 5)
plt.plot(years_future, y_prod_pred_future_tomatoes, label='Predicted Production (2025-2035)', color='red')
plt.title('Tomatoes - Production (Predicted 2025-2035)')
plt.xlabel('Year')
plt.ylabel('Production - Hundreds of thousands of tons')
plt.ylim(min(y_prod_pred_future_tomatoes)*0.9, max(y_prod_pred_future_tomatoes)*1.1)
for i in range(0, len(years_future), 2):  # Pas de 2 pentru a afisa din 2 in 2 ani
    plt.text(years_future[i], y_prod_pred_future_tomatoes[i], 
             f'{y_prod_pred_future_tomatoes[i]:.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')
plt.legend()

# Grafic linie pentru predicția Yield (roșii)
plt.subplot(2, 3, 6)
plt.plot(years_future, y_yield_pred_future_tomatoes, label='Predicted Yield (2025-2035)', color='red')
plt.title('Tomatoes - Yield (Predicted 2025-2035)')
plt.xlabel('Year')
plt.ylabel('Yield (kg/ha)')
plt.ylim(min(y_yield_pred_future_tomatoes)*0.99, max(y_yield_pred_future_tomatoes)*1.01)
for i in range(0, len(years_future), 2):  # Pas de 2 pentru a afisa din 2 in 2 ani
    plt.text(years_future[i], y_yield_pred_future_tomatoes[i], 
             f'{y_yield_pred_future_tomatoes[i]:.2f}', 
             ha='center', va='bottom', fontsize=10, color='black')
plt.legend()

plt.tight_layout()
plt.show()
