"""Pre-built SQL analytical queries for the dashboard."""
import sqlite3
import pandas as pd
import os

COST_PER_DOSE = 0.318  # USD
WASTAGE = 0.25

def get_queries() -> list[dict]:
    """Return list of dicts, each with 'name', 'description', 'sql'."""
    queries = [
        {
            'name': "Demand Growth by Country",
            'description': "Show year-over-year growth in MCV1_TARGET for each country, last 10 years",
            'sql': """
            WITH ranked AS (
                SELECT Country, Year, MCV1_TARGET,
                       LAG(MCV1_TARGET) OVER(PARTITION BY Country ORDER BY Year) as Prev_Target
                FROM vaccine_data
            )
            SELECT Country, Year, MCV1_TARGET,
                   ((MCV1_TARGET - Prev_Target) / Prev_Target) * 100 as Growth_Pct
            FROM ranked
            WHERE Year >= (SELECT MAX(Year) - 10 FROM vaccine_data)
            """
        },
        {
            'name': "Mortality Improvement Ranking",
            'description': "Rank countries by average annual decline in infant mortality rate over last 20 years",
            'sql': """
            WITH bounds AS (
                SELECT Country,
                       MIN(CASE WHEN Year = (SELECT MAX(Year) - 20 FROM vaccine_data) THEN "Infant Mortality Rate (infant deaths per 1,000 live births)" END) as start_val,
                       MIN(CASE WHEN Year = (SELECT MAX(Year) FROM vaccine_data) THEN "Infant Mortality Rate (infant deaths per 1,000 live births)" END) as end_val
                FROM vaccine_data
                WHERE Year IN ((SELECT MAX(Year) - 20 FROM vaccine_data), (SELECT MAX(Year) FROM vaccine_data))
                GROUP BY Country
            )
            SELECT Country, (start_val - end_val) / 20.0 as avg_annual_decline
            FROM bounds
            ORDER BY avg_annual_decline DESC
            """
        },
        {
            'name': "Demographic Snapshot",
            'description': "Latest year demographic comparison across all countries",
            'sql': """
            SELECT Country, Year, "Population, as of 1 July (thousands)", "Births (thousands)", 
                   "Infant Mortality Rate (infant deaths per 1,000 live births)", 
                   "Under-Five Mortality Rate (deaths under age 5 per 1,000 live births)", 
                   "Net Migration Rate (per 1,000 population)"
            FROM vaccine_data
            WHERE Year = (SELECT MAX(Year) FROM vaccine_data)
            """
        },
        {
            'name': "High Demand Years",
            'description': "Top 5 years with highest total demand across all countries",
            'sql': """
            SELECT Year, SUM(MCV1_TARGET) as Total_Demand
            FROM vaccine_data
            GROUP BY Year
            ORDER BY Total_Demand DESC
            LIMIT 5
            """
        },
        {
            'name': "Birth Rate Trends",
            'description': "Average birth rate by decade by country",
            'sql': """
            SELECT Country, 
                   (Year / 10) * 10 as Decade, 
                   AVG("Crude Birth Rate (births per 1,000 population)") as Avg_Birth_Rate
            FROM vaccine_data
            GROUP BY Country, Decade
            ORDER BY Country, Decade
            """
        },
        {
            'name': "Cost Projection",
            'description': "Estimated procurement cost per country per year (last 5 years)",
            'sql': """
            SELECT Country, Year, 
                   (MCV1_TARGET * 0.75 / (1 - 0.25) * 1000 * 0.318) as Est_Cost_USD
            FROM vaccine_data
            WHERE Year >= (SELECT MAX(Year) - 5 FROM vaccine_data)
            ORDER BY Country, Year
            """
        },
        {
            'name': "Demand Volatility",
            'description': "Standard deviation of year-over-year demand changes by country",
            'sql': """
            WITH changes AS (
                SELECT Country, Year,
                       MCV1_TARGET - LAG(MCV1_TARGET) OVER(PARTITION BY Country ORDER BY Year) as yoy_change
                FROM vaccine_data
            ),
            stats AS (
                SELECT Country, AVG(yoy_change) as avg_change
                FROM changes
                GROUP BY Country
            )
            SELECT c.Country, 
                   SQRT(AVG((c.yoy_change - s.avg_change) * (c.yoy_change - s.avg_change))) as demand_volatility
            FROM changes c
            JOIN stats s ON c.Country = s.Country
            WHERE c.yoy_change IS NOT NULL
            GROUP BY c.Country
            """
        },
        {
            'name': "Population vs Demand",
            'description': "Ratio of MCV1_TARGET to Pop_Age_0 by country by year",
            'sql': """
            SELECT Country, Year, MCV1_TARGET, "Pop_Age_0(In Thousands)",
                   (MCV1_TARGET * 1.0 / "Pop_Age_0(In Thousands)") as effective_coverage_proxy
            FROM vaccine_data
            WHERE "Pop_Age_0(In Thousands)" > 0
            """
        }
    ]
    return queries

def run_query(sql, db_path) -> pd.DataFrame:
    """Connect to SQLite db, execute query, return DataFrame."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)
