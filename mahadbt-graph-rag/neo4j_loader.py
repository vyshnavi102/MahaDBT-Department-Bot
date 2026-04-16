from neo4j import GraphDatabase
import pandas as pd
import psycopg2

# ------------------ DB CONFIG ------------------

conn = psycopg2.connect(
    host="",
    database="",
    user="",
    password="",
    port=""
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

# ------------------ NEO4J CONFIG ------------------
URI = "bolt://localhost:7687"
USERNAME = ""
PASSWORD = ""  

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# ------------------ INSERT FUNCTION ------------------
def insert_data(tx, row):
    tx.run("""
    MERGE (a:Application {applicationno: $applicationno})
    SET a.statusid = $statusid

    MERGE (s:Scheme {name: $schemename})
    MERGE (d:District {name: $districtname})
    MERGE (dept:Department {name: $departmentname})
    MERGE (fy:FinancialYear {year: $financialyear})

    MERGE (a)-[:BELONGS_TO_SCHEME]->(s)
    MERGE (a)-[:LOCATED_IN]->(d)
    MERGE (a)-[:UNDER_DEPARTMENT]->(dept)
    MERGE (a)-[:FOR_YEAR]->(fy)
    """, **row)

# ------------------ LOAD DATA ------------------
with driver.session() as session:
    for _, row in df.iterrows():
        session.execute_write(insert_data, row.to_dict())

print("Data loaded into Neo4j successfully!")
