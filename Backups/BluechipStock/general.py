import sqlite3
import pandas as pd

# Name of your database file
db_file = 'stock_data.db'

# Create a connection to the SQLite database
conn = sqlite3.connect(db_file)

# The table with the main data is 'price_history_NABIL'
table_name = 'price_history_NABIL'

# Use pandas to read the entire table into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Close the connection to the database
conn.close()

# Print the first 5 rows of the data
print(df.head())
print(df.shape)