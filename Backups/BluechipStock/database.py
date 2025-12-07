import sqlite3
import pandas as pd
import config

def init_db(symbol):
    """Creates a specific table for the given stock symbol if it doesn't exist."""
    table_name = f"price_history_{symbol}"
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                traded_quantity INTEGER NOT NULL,
                UNIQUE(symbol, date)
            )
        ''')
        conn.commit()
    print(f"Table '{table_name}' initialized successfully.")

def save_data(df, symbol):
    """Saves a DataFrame of new stock data to its specific table."""
    table_name = f"price_history_{symbol}"
    temp_table_name = f"temp_{table_name}"
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['published_date']).dt.date
    
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
        
        insert_sql = f'''
            INSERT OR IGNORE INTO {table_name} (symbol, date, open, high, low, close, traded_quantity)
            SELECT symbol, date, open, high, low, close, traded_quantity
            FROM {temp_table_name};
        '''
        conn.execute(insert_sql)
        conn.commit()

def get_last_date(symbol):
    """Gets the most recent date from a specific stock's table."""
    table_name = f"price_history_{symbol}"
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT MAX(date) FROM {table_name} WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()[0]
            return pd.to_datetime(result) if result else None
        except sqlite3.OperationalError:
            return None

def get_full_history(symbol):
    """Retrieves all historical data for a symbol from its specific table."""
    table_name = f"price_history_{symbol}"
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        query = f"SELECT * FROM {table_name} WHERE symbol = ? ORDER BY date ASC"
        df = pd.read_sql_query(query, conn, params=(symbol,))
        df['published_date'] = pd.to_datetime(df['date'])
        return df