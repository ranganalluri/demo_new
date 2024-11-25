from databricks import sql
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv("dev.env")


connection = None
cursor = None

try:
    
    connection = sql.connect(
        server_hostname=os.getenv("HOST_NAME"),
        http_path="/sql/1.0/warehouses/bfc0f97ed5a403c0",
        access_token=os.getenv("DATABRICKS_TOKEN"),
    )

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM mlflow_ops.default.taxi LIMIT 2")
    result = cursor.fetchall()

    # Get column names
    columns = [desc[0] for desc in cursor.description]

# Convert to DataFrame
    df = pd.DataFrame(result, columns=columns)
    print(df)
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if cursor is not None:
        cursor.close()
    if connection is not None:
        connection.close()
