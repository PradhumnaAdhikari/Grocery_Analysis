from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from visualizations import generate_forecast_image, generate_top_selling_products_images

app = Flask(__name__)

sales_model = joblib.load('grocery_model.pkl')
forecast_model = joblib.load('sales_forecast_model.pkl')
recommender_model = pd.read_csv('recommender_model.csv')
data = pd.read_excel('Online_Retail.xlsx')
product_list = data['Description'].unique().tolist()

@app.route('/')
def index():
    return render_template('index.html', products=product_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        description = request.form.get('Description', '').strip()
        quantity = request.form.get('Quantity', '').strip()
        unit_price = request.form.get('UnitPrice', '').strip()

        # Validate inputs
        if not description or not quantity or not unit_price:
            raise ValueError("Missing input values.")

        # Convert to float
        try:
            quantity = float(quantity)
            unit_price = float(unit_price)
        except ValueError:
            raise ValueError("Quantity and Unit Price must be valid numbers.")

        # Prepare input for model
        sales_input = pd.DataFrame([[quantity, unit_price]], columns=['Quantity', 'UnitPrice'])
        sales_prediction = sales_model.predict(sales_input)[0]

        # Calculate days to sell and purchase probability
        days_to_sell = calculate_days_to_sell(description, quantity)
        purchase_probability = calculate_purchase_probability(description, quantity)

        # Generate recommendations
        recommendations = get_recommendations(description)

        # Prepare result
        result = {
            'sales_prediction': sales_prediction,
            'unit_price': unit_price,
            'quantity': quantity,
            'product_description': description,
            'recommendations': recommendations,
            'days_to_sell': days_to_sell,
            'purchase_probability': purchase_probability
        }
        return jsonify(result)
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500


def calculate_days_to_sell(product_description, quantity):
    product_sales = data[data['Description'] == product_description].groupby('InvoiceDate')['Quantity'].sum().asfreq('D', fill_value=0)
    avg_daily_sales = product_sales.mean()
    if avg_daily_sales > 0:
        return round(quantity / avg_daily_sales, 2)
    else:
        return float('inf')

def calculate_purchase_probability(product_description, quantity):
    total_sales = data['Quantity'].sum()
    product_sales = data[data['Description'] == product_description]['Quantity'].sum()
    return round(product_sales / total_sales, 2) if total_sales > 0 else 0

def get_recommendations(product_description):
    # Check if recommender_model DataFrame is loaded correctly
    if recommender_model.empty:
        raise ValueError("Recommender model is empty or not loaded correctly.")
    
    # Apply eval safely
    try:
        recommender_model['antecedents'] = recommender_model['antecedents'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        recommender_model['consequents'] = recommender_model['consequents'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    except Exception as e:
        raise ValueError(f"Error applying eval: {str(e)}")
    
    # Filter rules where product_description is in antecedents
    rules = recommender_model[recommender_model['antecedents'].apply(lambda x: product_description in x)]
    
    # Debug output
    print(f"Filtered rules based on description '{product_description}': {rules}")

    # Sort rules by confidence and lift
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    recommendations = []
    for _, row in rules.head(5).iterrows():
        for item in row['consequents']:
            if item != product_description and item not in recommendations:
                recommendations.append(item)

    # Debug output
    print(f"Recommendations: {recommendations}")

    return recommendations


@app.route('/forecast')
def forecast():
    try:
        forecast_path = 'static/sales_forecast.png'
        generate_forecast_image(data, forecast_steps=30, image_path=forecast_path)
        generate_top_selling_products_images(data, 'static')

        return render_template('forecast.html', forecast_image=forecast_path)

    except Exception as e:
        print(f"Error generating forecast: {str(e)}")
        return jsonify({'error': f'Error generating forecast: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
