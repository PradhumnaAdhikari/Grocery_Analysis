import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import joblib

# Load the data
data = pd.read_excel('Online_Retail.xlsx')
basket = data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')

# Convert quantities to binary (1 if bought, 0 if not)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Find frequent itemsets and rules
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Save the rules
rules.to_csv('recommender_model.csv', index=False)
