import requests
import pandas as pd
from bs4 import BeautifulSoup
import config
import database

def _clean_and_convert_df(df):
    """A helper function to clean and format the dataframe."""
    df = df.rename(columns={'ltp': 'close', 'qty': 'traded_quantity'})
    columns_to_keep = ['published_date', 'open', 'high', 'low', 'close', 'traded_quantity']
    df = df[columns_to_keep]
    df['published_date'] = pd.to_datetime(df['published_date'])
    numeric_cols = ['open', 'high', 'low', 'close', 'traded_quantity']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df.dropna(inplace=True)
    return df

def _perform_full_scrape(session, payload):
    """Performs a full scrape of all historical data."""
    print("\nPerforming a one-time full historical data scrape...")
    all_data = []
    rows_per_page = 50
    for start_row in range(0, config.TOTAL_ROWS_TO_SCRAPE, rows_per_page):
        payload['start'] = str(start_row)
        payload['draw'] = str(int(start_row / rows_per_page) + 1)
        api_response = session.post("https://www.sharesansar.com/company-price-history", data=payload)
        if api_response.status_code != 200: break
        json_data = api_response.json()
        records = json_data.get('data', [])
        if not records: break
        all_data.extend(records)
        print(f"  - Scraped rows {start_row} - {start_row + len(records)}")
    if not all_data: return None
    df = pd.DataFrame(all_data)
    return _clean_and_convert_df(df)

def scrape_and_save_data(symbol, stock_info):
    """Scrapes data and returns True if new data was added, False otherwise."""
    payload = { 'draw': '1', 'columns[0][data]': 'DT_Row_Index', 'columns[0][name]': '', 'columns[0][searchable]': 'false', 'columns[0][orderable]': 'false', 'columns[0][search][value]': '', 'columns[0][search][regex]': 'false', 'columns[1][data]': 'published_date', 'columns[1][name]': '', 'columns[1][searchable]': 'true', 'columns[1][orderable]': 'true', 'columns[1][search][value]': '', 'columns[1][search][regex]': 'false', 'columns[2][data]': 'open', 'columns[2][name]': '', 'columns[2][searchable]': 'true', 'columns[2][orderable]': 'true', 'columns[2][search][value]': '', 'columns[2][search][regex]': 'false', 'columns[3][data]': 'high', 'columns[3][name]': '', 'columns[3][searchable]': 'true', 'columns[3][orderable]': 'true', 'columns[3][search][value]': '', 'columns[3][search][regex]': 'false', 'columns[4][data]': 'low', 'columns[4][name]': '', 'columns[4][searchable]': 'true', 'columns[4][orderable]': 'true', 'columns[4][search][value]': '', 'columns[4][search][regex]': 'false', 'columns[5][data]': 'ltp', 'columns[5][name]': '', 'columns[5][searchable]': 'true', 'columns[5][orderable]': 'true', 'columns[5][search][value]': '', 'columns[5][search][regex]': 'false', 'columns[6][data]': 'change', 'columns[6][name]': '', 'columns[6][searchable]': 'true', 'columns[6][orderable]': 'true', 'columns[6][search][value]': '', 'columns[6][search][regex]': 'false', 'columns[7][data]': 'qty', 'columns[7][name]': '', 'columns[7][searchable]': 'true', 'columns[7][orderable]': 'true', 'columns[7][search][value]': '', 'columns[7][search][regex]': 'false', 'columns[8][data]': 'turnover', 'columns[8][name]': '', 'columns[8][searchable]': 'true', 'columns[8][orderable]': 'true', 'columns[8][search][value]': '', 'columns[8][search][regex]': 'false', 'order[0][column]': '1', 'order[0][dir]': 'desc', 'start': '0', 'length': '50', 'search[value]': '', 'search[regex]': 'false', 'company': stock_info['company_id']}
    session = requests.Session()
    
    try:
        print(f"Visiting main page for {symbol} to get CSRF token...")
        response = session.get(stock_info['main_page_url'])
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        meta_tag = soup.find('meta', {'name': '_token'})
        if not meta_tag: raise ValueError("Could not find CSRF token.")
        
        csrf_token = meta_tag['content']
        session.headers.update({'X-CSRF-TOKEN': csrf_token, 'X-Requested-With': 'XMLHttpRequest'})

        last_saved_date = database.get_last_date(symbol)
        
        if last_saved_date is None:
            df_to_save = _perform_full_scrape(session, payload)
            if df_to_save is not None and not df_to_save.empty:
                database.save_data(df_to_save, symbol)
                print(f"\n🎉 Initial scrape for {symbol} complete! Total rows saved: {len(df_to_save)}")
                return True
        else:
            print(f"Checking for new data since {last_saved_date.strftime('%Y-%m-%d')}...")
            api_response = session.post("https://www.sharesansar.com/company-price-history", data=payload)
            if api_response.status_code != 200:
                print("Could not fetch new data from server.")
                return False
            
            records = api_response.json().get('data', [])
            if not records:
                print("No new data found on server.")
                return False
            
            new_df = pd.DataFrame(records)
            new_df = _clean_and_convert_df(new_df)
            
            brand_new_rows = new_df[new_df['published_date'] > last_saved_date]
            
            if not brand_new_rows.empty:
                print(f"Found {len(brand_new_rows)} new row(s) for {symbol}. Appending to database.")
                database.save_data(brand_new_rows, symbol)
                return True
            else:
                print(f"No new data found for {symbol}. Dataset is up-to-date.")
                return False
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"\nAn error occurred during scraping for {symbol}: {e}")
        return False