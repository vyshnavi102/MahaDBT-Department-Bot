import pandas as pd
import psycopg2  

conn = psycopg2.connect(
    host="",
    database="",
    user="",
    password=""
)

query = """
SELECT 
    applicationno,
    statusid,
    districtname,
    schemename,
    financialyear,
    departmentname
FROM dbt_dashboard."tbldashboard_2024-2025"
LIMIT 5000;
"""

df = pd.read_sql(query, conn)

print(df.head())
print("Total rows:", len(df))
