import pandas as pd
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

import pickle

data = pd.read_csv('mobile.csv')

features = ['Internal Memory', 'RAM', 'Rear Camera', 'Front Camera', 'No of Cameras', 'Battery']

y = data.Price
X = data[features]
X.head()
s = (X.dtypes == 'object')
object_cols = list(s[s].index)


label_X = X.copy()
ordinal_encoder = OrdinalEncoder()
label_X[object_cols] = ordinal_encoder.fit_transform(X[object_cols])
model = LinearRegression()

model.fit(label_X.values, y.values)

pickle.dump(model, open('model.pkl', 'wb'))
