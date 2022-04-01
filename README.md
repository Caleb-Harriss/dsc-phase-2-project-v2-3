![Real Estate Image](https://www.pexels.com/photo/house-lights-turned-on-106399/)
# Phase Two Project

For our phase two project we were tasked with creating a regression model for a real estate stakeholder.

We decided our stakeholder would be a man or woman looking for a house that will appreciate in value and is close to a middle school for their children.


# Data Understanding

* King County real estate data for homes sold in and around King County, Washington.
* Middle school locations in King County. We are able to calculate the distances from the houses in King County real estate data.
```
df = pd.read_csv('data/kc_house_data.csv')
df2 = pd.read_csv('data/middle_school_hd.csv')
df.info()
```
```
df.head()
```

## Middle School Distance Data 
We acquired this data through https://geo.wa.gov/datasets/23bbd746f9924c149681815cfd2a6300_0/about. We were able to filter this data to only select public middle schools in King County. We had the King County houses data and the middle school data on the same QGIS working map where we were able to use the 'distance to hub' geoprocessing tool. 

This tool calculated the distance from each of our houses to the nearest middle school. The middle school point locations were the 'Hubs', the processing tool ran and added two columns of data 'HubName' and 'HubDist' directly to our kc_houses_data attribute table (GIS dataframe). With these two new  columns I exported the layer as a csv and added it to our jupyter notebook for further cleaning and analysis.

```
df2.info()
```
```
df2.head()
```

# Data Preperation

This section goes over how the data was prepared to be used in our price prediction model.

## Data Cleaning 

NaN's were dropped, categorical variables were encoded and we found correlated variables with price

```
cleaned_df = df.dropna()
```
```
cleaned_df.sqft_basement.value_counts()
```
```
cleaned_df = cleaned_df.loc[cleaned_df.sqft_basement != '?']
cleaned_df.sqft_basement = cleaned_df.sqft_basement.astype('float')
```
```
cleaned_df2 = df2.dropna()
```


# Data Analysis

## Correlation Features with price
```
# import library
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')
```
```
# plot continuous variables with price
corr_df = abs(cleaned_df.corr().iloc[2:10])['price'].sort_values()
corr_df.plot.bar(figsize=(15,9), fontsize=20)
plt.title('Features Effect on Price', fontsize=20)
plt.xlabel('Features', fontsize=20)
plt.ylabel('Effect on price', fontsize=20)
plt.show()
```
```
# plot categorical variables with price except date

fig, axes2 = plt.subplots(nrows = 1, ncols = 4, figsize = (15, 10), sharey = True)

categoricals = ['waterfront', 'view', 'condition', 'grade']

for col, ax in zip(categoricals, axes2.flatten()):
    cleaned_df.groupby(col).mean()['price'].sort_values().plot.bar(ax=ax, fontsize=15)
    ax.set(xlabel=None)
    ax.set_title(col, fontsize=20)
    ax.set_yticks([])
    ax.set_ylabel('Price', fontsize=20)
fig.tight_layout()
```
# Feature Engineering

## Preprocessing Train Data
```
# preprocessing with scikit-learn
y = cleaned_df['price']
X = cleaned_df.drop('price', axis=1)
```
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```
```

print(f"X_train is a DataFrame with {X_train.shape[0]} rows and {X_train.shape[1]} columns")
print(f"y_train is a Series with {y_train.shape[0]} values")
```
```
# Select relevant Columns
relevant_columns = ['bathrooms',
                    'bedrooms',
                    'sqft_living',
                    'waterfront',
                    'view',
                    'condition',
                    'grade',
                    'sqft_basement',
                    'lat',
                    'floors',
                    'sqft_above'
                    ]

# Reassign X_train so that it only contains relevant columns
X_train = X_train.loc[:, relevant_columns]

```
```
X_train.isna().sum()
```
```
# Convert Categorical Features into Numbeers
X_train.info()
```
```
# date, waterfront, view, condition, and grade are objects
from sklearn.preprocessing import OrdinalEncoder
```
```
# waterfront transform
waterfront_train = X_train[['waterfront']]
encoder_waterfront = OrdinalEncoder(categories=[['NO', 'YES']])
encoder_waterfront.fit(waterfront_train)
waterfront_encoded_train = encoder_waterfront.transform(waterfront_train)
waterfront_encoded_train = waterfront_encoded_train.flatten()
X_train['waterfront'] = waterfront_encoded_train
```
```
# view transform
view_train = X_train[['view']]
encoder_view = OrdinalEncoder(categories=[['NONE', 'FAIR', 'AVERAGE', 'GOOD', 'EXCELLENT']])
encoder_view.fit(view_train)
view_encoded_train = encoder_view.transform(view_train)
view_encoded_train = view_encoded_train.flatten()
X_train['view'] = view_encoded_train
```
```
# condition transform
condition_train = X_train[['condition']]
encoder_condition = OrdinalEncoder(categories=[['Poor', 'Fair', 'Average', 'Good', 'Very Good']])
encoder_condition.fit(condition_train)
condition_encoded_train = encoder_condition.transform(condition_train)
condition_encoded_train = condition_encoded_train.flatten()
X_train['condition'] = condition_encoded_train
```
```
# grade transform
grade_train = X_train[['grade']]
encoder_grade = OrdinalEncoder(categories=[['3 Poor', '4 Low', '5 Fair', '6 Low Average', '7 Average', '8 Good', '9 Better', '10 Very Good', '11 Excellent', '12 Luxury', '13 Mansion']])
encoder_grade.fit(grade_train)
grade_encoded_train = encoder_grade.transform(grade_train)
grade_encoded_train = grade_encoded_train.flatten()
X_train['grade'] = grade_encoded_train
```
## Preprocess Test Data
```
# Drop Irrelevant Columns
X_test = X_test.loc[:, relevant_columns]

# Transform categorical values to numbers
# waterfront transform
waterfront_test = X_test[['waterfront']]
waterfront_encoded_test = encoder_waterfront.transform(waterfront_test).flatten()
X_test['waterfront'] = waterfront_encoded_test
# view transform
view_test = X_test[['view']]
view_encoded_test = encoder_view.transform(view_test).flatten()
X_test['view'] = view_encoded_test
# condition transform
condition_test = X_test[['condition']]
condition_encoded_test = encoder_condition.transform(condition_test).flatten()
X_test['condition'] = condition_encoded_test
# grade transform
grade_test = X_test[['grade']]
grade_encoded_test = encoder_grade.transform(grade_test).flatten()
X_test['grade'] = grade_encoded_test
```
## Find houses near middle school
```
# visualize the hub distance
cleaned_df2.describe()['HubDist'].loc[['25%','50%','75%']].plot.bar(figsize=(16,10), fontsize=20)
plt.axhline(0.7, color='red', label='0.7 miles', linewidth = 2)
plt.title('Top Three Interquartile Ranges for Home to School Distance', fontsize=20)
plt.ylabel('Distance')
plt.legend()
```
```
# drop houses' school disctance under 0.7 miles that is shown above
cleaned_df2 = cleaned_df2.loc[cleaned_df2.HubDist <= 0.7]
cleaned_df2.head()
```

# Modeling and Evaluation
```
# Model define
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```
```
# Evaluation with cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(model, X_train**2, y_train, cv=5)
```
```
# Evaluation with test set
model.fit(X_train**2, y_train)
model.score(X_test**2, y_test)
```
## House Price Prediction Modeling

# Prediction
```
# Get total data
y_total = pd.concat([y_test, y_train]).sort_index()
X_total = pd.concat([X_test, X_train]).sort_index()

# Predict price
pred = model.predict(X_total**2).round()
cleaned_df['predict_price'] = pred
```
```
# Visualize the real data with predicted data
average_df = cleaned_df[['price', 'predict_price']].mean()
average_df.plot.bar(fontsize=20, figsize=(10,9), color=['blue', 'orange'])
plt.title('Real Price VS Predicted Price', fontsize=20)
plt.xticks([0,1], ['Real Price: $541,497', 'Predicted: $542,798'], rotation=0)
plt.show()
```
## Apply to Business Problem
```
# Find houses that is under the 60% price that AI predicted
results_df = cleaned_df.loc[cleaned_df.price < cleaned_df.predict_price*0.6, ['id', 'price', 'predict_price', 'lat', 'long']]
results_df
```
```
# Visualize all houses
import folium

map1 = folium.Map(location=[47.5,-122])
points = (results_df.lat, results_df.long)
lat = points[0]
long = points[1]

for la, lo, real, pred in zip(lat, long, results_df.price, results_df.predict_price):
    iframe = folium.IFrame('price: ${} predict: ${}'.format(real, pred), width=100, height=100)
    popup = folium.Popup(iframe, max_width=100)
    folium.Marker(location=[la,lo],popup=popup).add_to(map1)
    
map1
```
```
# Available profits
results_df[['price', 'predict_price']].mean().plot.bar(figsize=(10,9), fontsize=20, color=['green', 'red'])
plt.xticks([0,1],['BUY: $364,219', 'VALUE: $698,198'], rotation=0);
plt.title('Profitable Houses Average Price', fontsize=20)
```
# Visualize houses that is near middle school
results_df = results_df.join(cleaned_df2, how='inner', lsuffix='index')
map2 = folium.Map(location=[47.5,-122])
points = (results_df.lat, results_df.long)
lat = points[0]
long = points[1]

for la, lo, real, pred in zip(lat, long, results_df.price, results_df.predict_price):
    iframe = folium.IFrame('price: ${} predict: ${}'.format(real, pred), width=100, height=100)
    popup = folium.Popup(iframe, max_width=100)
    folium.Marker(location=[la,lo],popup=popup).add_to(map2)
```
```
map2
```
# Available profits
results_df[['price', 'predict_price']].mean().plot.bar(figsize=(10,9), fontsize=20, color=['blue', 'red'])
plt.xticks([0,1],['BUY: $301,645', 'VALUE: $564,435'], rotation=0);
plt.title('Profitable Houses Near Middle School', fontsize=20)
```
# Conclusion 
* 71.5% of the data fit our house price prediction model.

* The model was able to recommend 507 houses to purchase after finding homes where the actual price was 40% lower than the predicted price.

* To mitigate the commute time for the middle school child we found how many of the 507 houses fall within 0.7 miles from the closest middle school. We found a final list of100 houses that lie within 0.7 miles from a middle school!

