import pandas as pd
#import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

model = tf.keras.models.load_model('/workspaces/Coffee-shope-profit-prediction-app/Coffee shope prediction Model2.h5')

test_data= pd.read_csv("cleaned_data.csv")

x_test= test_data[['unit_price', 'transaction_qty', 'Total Sales']]
y_test= test_data['Profit']

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(x_test)
y_test_scaled= scaler.fit_transform(y_test.values.reshape(-1,1))
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test_scaled,y_pred)
print(mse)
#with open('accuracy.txt','w') as file:
    #file.write(f'Accuracy: {mse}')