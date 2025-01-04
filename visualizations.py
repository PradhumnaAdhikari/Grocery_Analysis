import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def generate_forecast_image(data, forecast_steps, image_path):
    # Drop rows with missing 'InvoiceDate' or 'Quantity'
    data.dropna(subset=['InvoiceDate', 'Quantity'], inplace=True)
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data = data[data['Quantity'] > 0]
    daily_sales = data.groupby('InvoiceDate')['Quantity'].sum().asfreq('D', fill_value=0)

    # Fit ARIMA model
    model = ARIMA(daily_sales, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast future sales
    forecast = model_fit.forecast(steps=forecast_steps)

    # Generate date range for forecasted values
    forecast_index = pd.date_range(start=daily_sales.index.max() + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_df = pd.DataFrame(data={'Quantity': forecast}, index=forecast_index)

    # Plot forecast
    plt.figure(figsize=(14, 7))
    plt.plot(daily_sales, label='Historical Sales', color='blue')
    plt.plot(forecast_df.index, forecast_df['Quantity'], color='red', linestyle='--', label='Forecasted Sales')
    plt.title('Sales Forecast for Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def generate_top_selling_products_images(data, static_folder):
    try:
        # Process data for top-selling products
        daily_sales = data.groupby('InvoiceDate')['Quantity'].sum().asfreq('D', fill_value=0)

        # Create a bar chart for top-selling products
        top_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        top_products.plot(kind='bar')
        plt.title('Top 10 Selling Products')
        plt.xlabel('Product')
        plt.ylabel('Total Quantity Sold')
        plt.tight_layout()
        plt.savefig(f'{static_folder}/top_10_products.png')
        plt.close()

        # Create a pie chart for top-selling products
        plt.figure(figsize=(8, 8))
        plt.pie(top_products, labels=top_products.index, autopct='%1.1f%%', startangle=140)
        plt.title('Sales Distribution of Top 10 Products')
        plt.tight_layout()
        plt.savefig(f'{static_folder}/top_10_products_pie.png')
        plt.close()

    except Exception as e:
        print(f"Error generating top-selling products images: {e}")
