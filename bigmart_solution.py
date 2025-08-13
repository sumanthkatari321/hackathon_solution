
# BigMart Sales Prediction Script

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Keep a copy of IDs for submission
test_ids = test[["Item_Identifier", "Outlet_Identifier"]]

# Combine train and test for preprocessing
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True)

# Fill missing Item_Weight
item_weight_mean = data.pivot_table(values='Item_Weight', index='Item_Identifier')
def impute_weight(row):
    if pd.isnull(row['Item_Weight']):
        return item_weight_mean.loc[row['Item_Identifier']].values[0]
    else:
        return row['Item_Weight']
data['Item_Weight'] = data.apply(impute_weight, axis=1)
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

# Fill missing Outlet_Size
outlet_size_mode = data.pivot_table(values='Outlet_Size', index='Outlet_Type', aggfunc=lambda x: x.mode()[0])
def impute_outlet_size(row):
    if pd.isnull(row['Outlet_Size']):
        return outlet_size_mode.loc[row['Outlet_Type']].values[0]
    else:
        return row['Outlet_Size']
data['Outlet_Size'] = data.apply(impute_outlet_size, axis=1)

# Standardize Item_Fat_Content
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

# Feature engineering
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

# Log transform visibility
data['Item_Visibility'] = data['Item_Visibility'].replace(0, np.nan)
data['Item_Visibility'].fillna(data['Item_Visibility'].mean(), inplace=True)
data['Item_Visibility_Log'] = np.log1p(data['Item_Visibility'])

# Encode categorical variables
le = LabelEncoder()
cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    if col not in ['source', 'Item_Identifier', 'Outlet_Identifier']:
        data[col] = le.fit_transform(data[col])

# Split back
train_clean = data[data['source'] == 'train'].drop(['source'], axis=1)
test_clean = data[data['source'] == 'test'].drop(['source', 'Item_Outlet_Sales'], axis=1)

X = train_clean.drop(['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'], axis=1)
y = train_clean['Item_Outlet_Sales']

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X, y)

# Predict
predictions = model.predict(test_clean.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1))

# Create submission
submission = test_ids.copy()
submission['Item_Outlet_Sales'] = predictions
submission.to_csv("submission.csv", index=False)
