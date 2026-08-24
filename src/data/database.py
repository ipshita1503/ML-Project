"""SQLite database for SQL analytics demonstrations."""
import sqlite3
import os
import pandas as pd

def create_database(df_historical: pd.DataFrame, df_future: pd.DataFrame, db_path: str = None) -> str:
    """Create SQLite database and populate with data."""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'giip.db')
        
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Demographics table
    demographics_cols = ['Year', 'Country', 'Total Population, as of 1 January (thousands)', 
                         'Births (thousands)', 'Infant Deaths, under age 1 (thousands)', 
                         'Net Number of Migrants (thousands)']
    
    all_df = pd.concat([df_historical, df_future], ignore_index=True)
    demographics_df = all_df[demographics_cols].rename(columns={
        'Total Population, as of 1 January (thousands)': 'population',
        'Births (thousands)': 'births',
        'Infant Deaths, under age 1 (thousands)': 'mortality',
        'Net Number of Migrants (thousands)': 'migration'
    })
    demographics_df.to_sql('demographics', conn, if_exists='replace', index=False)
    
    # Vaccine targets
    targets_df = df_historical[['Year', 'Country', 'MCV1_TARGET']]
    targets_df.to_sql('vaccine_targets', conn, if_exists='replace', index=False)
    
    # Forecasts (empty for now)
    conn.execute("CREATE TABLE IF NOT EXISTS forecasts (Year INTEGER, Country TEXT, forecasted_demand REAL)")
    
    conn.close()
    return db_path

def run_query(query: str, db_path: str) -> pd.DataFrame:
    """Execute SQL query and return results."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_analytical_queries() -> list[dict]:
    """Return a list of pre-built analytical queries."""
    return [
        {
            'name': 'Year-over-year demand growth by country',
            'description': 'Calculates the year-over-year percentage growth in vaccine demand for each country.',
            'sql': '''
            SELECT t1.Country, t1.Year, 
                   (t1.MCV1_TARGET - t2.MCV1_TARGET) / t2.MCV1_TARGET * 100 as YoY_Growth
            FROM vaccine_targets t1
            JOIN vaccine_targets t2 ON t1.Country = t2.Country AND t1.Year = t2.Year + 1
            '''
        },
        {
            'name': 'Countries ranked by average infant mortality decline',
            'description': 'Ranks countries by the average decline in infant mortality.',
            'sql': '''
            SELECT Country, AVG(mortality_decline) as avg_decline
            FROM (
                SELECT t1.Country, (t2.mortality - t1.mortality) as mortality_decline
                FROM demographics t1
                JOIN demographics t2 ON t1.Country = t2.Country AND t1.Year = t2.Year + 1
            )
            GROUP BY Country
            ORDER BY avg_decline ASC
            '''
        },
        {
            'name': 'Demographic snapshot comparison across countries (latest year)',
            'description': 'Compares key demographic metrics for the latest year.',
            'sql': '''
            SELECT Country, population, births, mortality, migration
            FROM demographics
            WHERE Year = (SELECT MAX(Year) FROM demographics)
            '''
        },
        {
            'name': 'Correlation between birth rate changes and demand changes',
            'description': 'Shows changes in births and demand to analyze correlation.',
            'sql': '''
            SELECT d1.Country, d1.Year, 
                   (d1.births - d2.births) as birth_change,
                   (t1.MCV1_TARGET - t2.MCV1_TARGET) as demand_change
            FROM demographics d1
            JOIN demographics d2 ON d1.Country = d2.Country AND d1.Year = d2.Year + 1
            JOIN vaccine_targets t1 ON d1.Country = t1.Country AND d1.Year = t1.Year
            JOIN vaccine_targets t2 ON d1.Country = t2.Country AND d2.Year = t2.Year
            '''
        },
        {
            'name': 'Peak demand years by country',
            'description': 'Identifies the year with the highest vaccine demand for each country.',
            'sql': '''
            SELECT Country, Year, MAX(MCV1_TARGET) as peak_demand
            FROM vaccine_targets
            GROUP BY Country
            '''
        },
        {
            'name': 'Decade-over-decade demographic shift analysis',
            'description': 'Analyzes average population changes per decade.',
            'sql': '''
            SELECT Country, (Year / 10) * 10 as Decade, AVG(population) as avg_population
            FROM demographics
            GROUP BY Country, Decade
            ORDER BY Country, Decade
            '''
        }
    ]
