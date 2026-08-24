"""Feature engineering for vaccine demand forecasting."""
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, target_col: str = 'MCV1_TARGET') -> tuple[pd.DataFrame, list[str]]:
    """
    Engineer features for demand forecasting.
    
    CRITICAL: All target-derived features MUST use .shift() to prevent data leakage.
    Lagged target features look at past data so that no future information is used to predict the current year.
    """
    df = df.copy()
    df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
    
    # 1. Raw features
    raw_cols = ['Year', 'Total Population, as of 1 January (thousands)', 'Births (thousands)', 
                'Crude Birth Rate (births per 1,000 population)', 'Infant Deaths, under age 1 (thousands)', 
                'Infant Mortality Rate (infant deaths per 1,000 live births)', 'Net Migration Rate (per 1,000 population)', 
                'Pop_Age_0(In Thousands)']
                
    # 2. Derived ratios
    df['Births_per_1000pop'] = df['Births (thousands)'] / df['Total Population, as of 1 July (thousands)'] * 1000
    df['Infant_mort_ratio'] = df['Infant Deaths, under age 1 (thousands)'] / df['Births (thousands)']
    df['U5_mort_ratio'] = df['Under-Five Deaths, under age 5 (thousands)'] / df['Births (thousands)']
    
    # 3. Target history (preventing leakage using shift)
    if target_col in df.columns:
        df['target_lag_3'] = df.groupby('Country')[target_col].shift(3)
        # Using shift(1) to avoid including current row in the rolling mean
        df['target_roll3_mean'] = df.groupby('Country')[target_col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    else:
        df['target_lag_3'] = np.nan
        df['target_roll3_mean'] = np.nan
        
    # 4. Lagged demographic changes
    df['Births_YoY_growth'] = df.groupby('Country')['Births (thousands)'].transform(lambda x: x.pct_change().shift(1))
    df['Population_YoY_growth'] = df.groupby('Country')['Total Population, as of 1 January (thousands)'].transform(lambda x: x.pct_change().shift(1))
    df['BirthRate_change'] = df.groupby('Country')['Crude Birth Rate (births per 1,000 population)'].transform(lambda x: x.diff().shift(1))
    df['InfantMortality_change'] = df.groupby('Country')['Infant Mortality Rate (infant deaths per 1,000 live births)'].transform(lambda x: x.diff().shift(1))
    
    # 5. Country dummies (Kyrgyzstan is reference)
    df['Country_Lesotho'] = (df['Country'] == 'Lesotho').astype(int)
    df['Country_Uzbekistan'] = (df['Country'] == 'Uzbekistan').astype(int)
    
    # Drop rows where critical lags are NaN
    df = df.dropna(subset=['target_lag_3', 'Births_YoY_growth']).reset_index(drop=True)
    
    feature_cols = raw_cols + [
        'Births_per_1000pop', 'Infant_mort_ratio', 'U5_mort_ratio', 
        'target_lag_3', 'target_roll3_mean', 
        'Births_YoY_growth', 'Population_YoY_growth', 'BirthRate_change', 'InfantMortality_change',
        'Country_Lesotho', 'Country_Uzbekistan'
    ]
    
    return df, feature_cols

def get_feature_descriptions() -> dict[str, str]:
    """Return human-readable descriptions for features."""
    return {
        'Year': 'The year of the observation',
        'Total Population, as of 1 January (thousands)': 'Total population at start of year',
        'Births (thousands)': 'Number of births in thousands',
        'Crude Birth Rate (births per 1,000 population)': 'Crude birth rate per 1,000 people',
        'Infant Deaths, under age 1 (thousands)': 'Number of infant deaths',
        'Infant Mortality Rate (infant deaths per 1,000 live births)': 'Infant mortality rate',
        'Net Migration Rate (per 1,000 population)': 'Net migration rate',
        'Pop_Age_0(In Thousands)': 'Population of age 0',
        'Births_per_1000pop': 'Ratio of births per 1,000 mid-year population',
        'Infant_mort_ratio': 'Ratio of infant deaths to births',
        'U5_mort_ratio': 'Ratio of under-5 deaths to births',
        'target_lag_3': 'Vaccine demand 3 years prior (prevents leakage)',
        'target_roll3_mean': 'Rolling 3-year mean of vaccine demand, shifted by 1 year',
        'Births_YoY_growth': 'Year-over-year growth in births, lagged by 1 year',
        'Population_YoY_growth': 'Year-over-year growth in population, lagged by 1 year',
        'BirthRate_change': 'Annual change in crude birth rate, lagged by 1 year',
        'InfantMortality_change': 'Annual change in infant mortality rate, lagged by 1 year',
        'Country_Lesotho': 'Dummy variable for Lesotho',
        'Country_Uzbekistan': 'Dummy variable for Uzbekistan'
    }
