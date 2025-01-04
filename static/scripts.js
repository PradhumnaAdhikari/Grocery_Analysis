document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('predict-form').addEventListener('submit', function(event) {
        event.preventDefault();
        document.getElementById('predict-form').style.display = 'none'; 
        document.getElementById('results').style.display = 'none';
    
        
        let formData = new FormData(this);
    
        fetch('/predict', {
            method: 'POST',
            body: new URLSearchParams(formData)
        })
        .then(response => response.json())
        .then(data => {
            
            document.getElementById('predict-form').style.display = 'block';
            if (data.error) {
                alert(data.error);
            } else {
                document.getElementById('results').style.display = 'block';
                document.getElementById('profit-result').innerText = `Profit Prediction: ${data.sales_prediction}`;
                document.getElementById('recommendations-list').innerHTML = data.recommendations.map(item => `<li>${item}</li>`).join('');
                document.getElementById('days-to-sell').innerText = `Days to Sell: ${data.days_to_sell}`;
                document.getElementById('purchase-probability').innerText = `Purchase Probability: ${data.purchase_probability}`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('predict-form').style.display = 'block';
        });
    });
})    