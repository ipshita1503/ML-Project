"""Data loading and validation for the GIIP pipeline."""
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

TARGET = 'MCV1_TARGET'
COUNTRIES = ['Kyrgyzstan', 'Lesotho', 'Uzbekistan']

EXPECTED_COLUMNS = [
    'Country', 'Year',
    'Total Population, as of 1 January (thousands)',
    'Total Population, as of 1 July (thousands)',
    'Births (thousands)',
    'Crude Birth Rate (births per 1,000 population)',
    'Infant Deaths, under age 1 (thousands)',
    'Infant Mortality Rate (infant deaths per 1,000 live births)',
    'Under-Five Deaths, under age 5 (thousands)',
    'Under-Five Mortality (deaths under age 5 per 1,000 live births)',
    'Net Number of Migrants (thousands)',
    'Net Migration Rate (per 1,000 population)',
    'Pop_Age_0(In Thousands)',
]

def load_vaccine_data() -> pd.DataFrame:
    """Load vaccine historical data and validate."""
    file_path = os.path.join(DATA_DIR, 'vaccine_data.csv')
    df = pd.read_csv(file_path)
    
    # Validation
    missing_cols = [col for col in EXPECTED_COLUMNS + [TARGET] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
        
    if (df['Total Population, as of 1 January (thousands)'] < 0).any():
        raise ValueError("Negative populations found.")
        
    print(f"Data loaded from {df['Year'].min()} to {df['Year'].max()}")
    print("Summary stats:\n", df.describe())
    
    return df.sort_values(['Country', 'Year']).reset_index(drop=True)

def load_future_demographics() -> pd.DataFrame:
    """Load future demographics data and validate."""
    file_path = os.path.join(DATA_DIR, 'future_demographics.csv')
    df = pd.read_csv(file_path)
    
    # Validation
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
        
    return df.sort_values(['Country', 'Year']).reset_index(drop=True)

def validate_data(df: pd.DataFrame) -> dict:
    """Validate data and return summary dictionary."""
    summary = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'year_range': (int(df['Year'].min()), int(df['Year'].max())),
        'countries': df['Country'].unique().tolist(),
        'row_count': len(df)
    }
    
    print("Data Validation Summary:")
    print(f"Rows: {summary['row_count']}")
    print(f"Duplicates: {summary['duplicates']}")
    print(f"Years: {summary['year_range'][0]} - {summary['year_range'][1]}")
    print(f"Countries: {summary['countries']}")
    
    return summary

def get_data_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics for dashboard."""
    return df.describe().to_dict()
