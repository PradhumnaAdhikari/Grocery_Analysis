import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the data from Excel
data = pd.read_excel('Online_Retail.xlsx')

# Display columns to confirm they are present
print("Columns in the data:", data.columns)

# Drop rows with missing 'InvoiceDate' or 'Quantity'
data.dropna(subset=['InvoiceDate', 'Quantity'], inplace=True)

# Convert 'InvoiceDate' to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Filter out transactions with non-positive quantities
data = data[data['Quantity'] > 0]

# Aggregate data by date
daily_sales = data.groupby('InvoiceDate')['Quantity'].sum()

# Ensure the index has a frequency (daily frequency)
daily_sales = daily_sales.asfreq('D', fill_value=0)

# Check the data
print("Daily Sales Data:")
print(daily_sales.head())
print("Index:")
print(daily_sales.index)
print("Frequency:")
print(daily_sales.index.freq)
print("Data Type and Shape:")
print(daily_sales.dtype)
print(daily_sales.shape)

# Fit ARIMA model with specified parameters (p, d, q)
# Adjust these parameters as needed
model = ARIMA(daily_sales, order=(5, 1, 0))  # Adjust order as needed

try:
    # Fit the model
    model_fit = model.fit()
    print(model_fit.summary())

    # Forecast future sales (e.g., next 30 days)
    forecast_steps = 15
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate date range for forecasted values
    forecast_index = pd.date_range(start=daily_sales.index.max() + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_df = pd.DataFrame(data={'Quantity': forecast}, index=forecast_index)

    # Plot forecast
    plt.figure(figsize=(14, 7))  # Adjust the figure size for better visibility
    plt.plot(daily_sales, label='Historical Sales', color='blue')
    plt.plot(forecast_df.index, forecast_df['Quantity'], color='red', linestyle='--', label='Forecasted Sales')
    plt.title('Sales Forecast for Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig('static/sales_forecast.png')
    plt.close()  # Close the plot to free memory

except Exception as e:
    print(f"Error in ARIMA model fitting or forecasting: {e}")
