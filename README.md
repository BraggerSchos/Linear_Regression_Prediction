# Linear_Regression_Prediction  
Cabbages and tomatoes production prediction  

This Python script is designed to perform data analysis and predictions on cabbage and tomato crops based on historical data. The script uses various machine learning techniques, including linear regression models, to predict future yields, area harvested, and production for both crops from the years 2025 to 2035. Additionally, it generates visualizations for both real and predicted data trends.  

## 1. Requirements  
Before running the script, ensure that the following Python libraries are installed:  
- pandas  
- numpy  
- matplotlib  
- scikit-learn  

These can be installed using pip:  
```bash
pip install pandas numpy matplotlib scikit-lear
```

## 2. Input data  
The script expects a CSV file containing agricultural data. Ensure that the data includes columns for:  
- Year  
- Item (which should contain "Cabbages" or "Tomatoes")  
- Area Harvested (ha)  
- Production (t)  
- Yield (kg/ha)  

## 3. Script steps  

**Load data**: The script loads the data from the specified CSV file using the pandas.read_csv function.  
**Filter data**: It separates the data into two subsets for cabbage and tomato crops.  
**Preprocess data**: The Year column is used as the feature (X) and Area Harvested, Production, and Yield are used as target variables (y).  
**Split data**: The dataset is split into training and testing sets, with 20% of the data reserved for testing.  
**Standardize data**: The features are standardized using StandardScaler to improve the performance of the linear regression models.  
**Train models**: Three linear regression models are created and trained for each crop:  
  - Area Harvested prediction  
  - Production prediction  
  - Yield prediction  

**Make Predictions**: Predictions are made for both the cabbage and tomato crops on the test data and for future years (2025-2035).  
**Visualizations**: The script generates line plots for both real and predicted data, including:  
  - Area Harvested  
  - Production  
  - Yield  

The plots are displayed using matplotlib.  

## 4. Output  
**Predicted data**: The script predicts values for the years 2025-2035 and visualizes them alongside the real data.  
**Visual plots**: The results are plotted for both cabbage and tomato crops, showing trends over time and the predicted values for future years.  

## 5. Example plots  
- Area Harvested (Real vs Predicted) for both cabbage and tomato crops.  
- Production (Real vs Predicted) for both crops.  
- Yield (Real vs Predicted) for both crops.  

## 6. Data sources  
The data used in athis script was sourced from FAOSTAT (Food and Agriculture Organization of the United Nations) website. Please make sure to credit the source if you use the data for any research or analysis.  


## 7. License  
This script is free to use for personal and commercial purposes. You are free to modify and distribute it as you wish.  
