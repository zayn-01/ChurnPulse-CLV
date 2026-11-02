# ChurnPulse-CLV

An end-to-end customer analytics pipeline built for retail e-commerce. I built this project to move beyond basic classification models and explore probabilistic modeling for non-contractual customer behavior. It uses the UCI Online Retail II dataset to predict customer lifetime value (CLV), churn risk, and future purchase frequency using BG/NBD and Gamma-Gamma models.

The project is structured as a modular Python application featuring a clean ETL pipeline, a FastAPI REST backend for real-time scoring, and an interactive Streamlit dashboard for business stakeholders.

## Repository Structure

```
customer-clv-pipeline/
├── api/
│   └── main.py              # FastAPI backend serving real-time predictions
├── app/
│   └── dashboard.py         # Streamlit executive UI & prescriptive engine
├── src/
│   ├── models/
│   │   └── clv_bgnbd.py     # Wrapper class for BG/NBD & Gamma-Gamma models
│   └── data_pipeline.py     # Robust data cleaning, preprocessing & RFM conversion
├── online_retail_II.csv     # UCI transactional dataset
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

## Technical Approach

1. **Data Ingestion & Hygiene (`data_pipeline.py`):**
   * Loads the raw retail transactional dataset, dropping unassigned customer records.
   * Cleans out invoice cancellations (refunds starting with `'C'`, non-positive quantities, and zero/negative prices).
   * Filters out non-product administrative stock codes (like `POST` postage fees and manual adjustments) so monetary values are not distorted.
   * Aggregates line items into unique checkout baskets and transforms raw timestamps into RFM (Recency, Frequency, T, Monetary) feature vectors.

2. **Probabilistic Modeling (`clv_bgnbd.py`):**
   * **BG/NBD Model:** Predicts repeat purchase frequency and estimates the probability that a customer is still active versus quietly churned.
   * **Gamma-Gamma Model:** Estimates average monetary value per customer independently of transaction frequency to project long-term customer value.

3. **Backend API (`api/main.py`):**
   * Built with FastAPI and Pydantic to validate incoming customer metrics payloads and return real-time 90-day purchase and churn predictions.

4. **Executive Dashboard (`app/dashboard.py`):**
   * An interactive Streamlit cockpit displaying high-level metrics, a value versus churn risk scatter plot (Plotly), and an automated recommendation engine for customer retention.

## Getting Started Locally

### 1. Clone and Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Server
```bash
uvicorn api.main:app --reload
```
The API documentation will be live at `http://127.0.0.1:8000/docs`.

### 4. Run the Streamlit Dashboard
Open a separate terminal window, activate your virtual environment, and launch the app:
```bash
streamlit run app/dashboard.py
```

## Tech Stack

* Python 3.10+
* FastAPI & Uvicorn (API Backend)
* Streamlit & Plotly (Interactive UI & Visualizations)
* Lifetimes & Scikit-Learn (Probabilistic ML Models)
* Pandas & NumPy (ETL & Data Wrangling)