import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_real_retail_data(filename="online_retail_II.csv"):
    """Loads and cleans the real UCI Online Retail II dataset with safe path resolution and advanced data hygiene."""
    # Look for the file in the current directory or a 'data' folder
    possible_paths = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename),
        os.path.join("..", "data", filename)
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if not file_path:
        raise FileNotFoundError(
            f"Could not find {filename}. Please place the CSV file in your project root or a 'data/' folder."
        )
    
    # Read the CSV file directly
    df = pd.read_csv(file_path)
    
    # 1. Drop rows without a Customer ID
    df = df.dropna(subset=['Customer ID'])
    
    # 2. Filter out cancellations (Invoice starting with 'C' or Quantity <= 0)
    df['Invoice'] = df['Invoice'].astype(str)
    df = df[~df['Invoice'].str.startswith('C')]
    df = df[df['Quantity'] > 0]
    
    # 3. Filter out zero or negative prices (pricing errors or free items)
    df = df[df['Price'] > 0]
    
    # 4. Filter out non-product administrative stock codes (postage, manual adjustments, bank charges)
    non_product_codes = ['POST', 'DOT', 'M', 'BANK CHARGES', 'C2', 'PADS', 'AMAZONFEE']
    df = df[~df['StockCode'].isin(non_product_codes)]
    
    # 5. Convert timestamps
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 6. Calculate monetary value per line item
    df['order_value'] = df['Quantity'] * df['Price']
    
    # 7. Aggregate line items into unique orders per customer
    order_df = df.groupby(['Customer ID', 'Invoice']).agg(
        order_date=('InvoiceDate', 'min'),
        order_value=('order_value', 'sum')
    ).reset_index()
    
    order_df = order_df.rename(columns={'Customer ID': 'customer_id'})
    return order_df.sort_values("order_date").reset_index(drop=True)

def generate_retail_transactions(num_customers=500, days=365):
    """Fallback synthetic e-commerce transactional dataset generator."""
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=days)
    records = []
    
    for customer_id in range(1001, 1001 + num_customers):
        p_buy = np.random.beta(2, 5)
        num_orders = np.random.poisson(lam=p_buy * 20) + 1
        
        for _ in range(num_orders):
            days_offset = np.random.randint(0, days)
            order_date = start_date + timedelta(days=int(days_offset))
            monetary_val = round(np.random.gamma(shape=3, scale=25), 2)
            records.append({
                "customer_id": customer_id,
                "order_date": order_date,
                "order_value": monetary_val
            })
            
    df = pd.DataFrame(records)
    return df.sort_values("order_date").reset_index(drop=True)

def transform_to_rfm(df, observation_date=None):
    """Converts raw transactions into RFM summary format for BG/NBD models."""
    if observation_date is None:
        observation_date = df['order_date'].max()
    
    rfm = df.groupby('customer_id').agg(
        frequency=('order_date', lambda x: len(x) - 1),  # Repeat purchases
        recency=('order_date', lambda x: (x.max() - x.min()).days),
        T=('order_date', lambda x: (observation_date - x.min()).days),
        monetary_value=('order_value', lambda x: x.iloc[1:].mean() if len(x) > 1 else 0)
    ).reset_index()
    
    rfm = rfm[rfm['T'] > 0]
    return rfm

if __name__ == "__main__":
    df_raw = load_real_retail_data()
    rfm_df = transform_to_rfm(df_raw)
    print("Real RFM Sample Loaded Successfully:")
    print(rfm_df.head())