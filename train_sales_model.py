import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Load the data
data = pd.read_excel('Online_Retail.xlsx')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
daily_sales = data.groupby('InvoiceDate')['Quantity'].sum().asfreq('D', fill_value=0)

# Fit ARIMA model
model = ARIMA(daily_sales, order=(1, 1, 1))
model_fit = model.fit()

# Save the model
joblib.dump(model_fit, 'sales_forecast_model.pkl')
