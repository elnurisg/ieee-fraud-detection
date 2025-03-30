import pandas as pd
import numpy as np
import datetime as dt

def transform_transaction_dt(df, ref_date="2017-11-30"):
    startdate = dt.datetime.strptime(ref_date, '%Y-%m-%d')
    df['date'] = df['TransactionDT'].apply(lambda x: (startdate + dt.timedelta(seconds = x)))
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df["date"] = df['date'].dt.date
    
    return df

# Define the main providers
main_email_providers = {"gmail.com", "yahoo.com", "hotmail.com"}

def map_email_provider(domain):
    if pd.isnull(domain):
        return "unknown"
    domain = domain.lower()
    return domain if domain in main_email_providers else "other"