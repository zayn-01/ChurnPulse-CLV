from lifetimes import BetaGeoFitter, GammaGammaFitter
import pandas as pd

class RetailCLVPipeline:
    def __init__(self, penalizer_coef=0.01):
        self.bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        self.ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
        
    def fit(self, rfm_df):
        """Fits BG/NBD and Gamma-Gamma models on RFM summary data."""
        # Fit BG/NBD for transaction frequency prediction
        self.bgf.fit(rfm_df['frequency'], rfm_df['recency'], rfm_df['T'])
        
        # Fit Gamma-Gamma for average monetary value (only repeat buyers)
        repeat_customers = rfm_df[rfm_df['frequency'] > 0]
        self.ggf.fit(repeat_customers['frequency'], repeat_customers['monetary_value'])
        
    def predict(self, rfm_df, time_horizon_days=90, discount_rate=0.01):
        """Predicts expected purchases, survival prob, and CLV over horizon."""
        df_pred = rfm_df.copy()
        
        # Expected repeat purchases in next N days
        df_pred['predicted_purchases'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
            time_horizon_days, df_pred['frequency'], df_pred['recency'], df_pred['T']
        )
        
        # Probability of being active (1 - Churn Probability)
        df_pred['p_active'] = self.bgf.conditional_probability_alive(
            df_pred['frequency'], df_pred['recency'], df_pred['T']
        )
        df_pred['p_churn'] = 1.0 - df_pred['p_active']
        
        # Predicted Customer Lifetime Value (CLV)
        df_pred['predicted_clv'] = self.ggf.customer_lifetime_value(
            self.bgf,
            df_pred['frequency'],
            df_pred['recency'],
            df_pred['T'],
            df_pred['monetary_value'],
            time=int(time_horizon_days / 30), # in months
            discount_rate=discount_rate
        )
        
        return df_pred