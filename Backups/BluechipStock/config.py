# --- Master list of stocks to track ---
STOCKS = {
    'NABIL': {'company_id': '16', 'main_page_url': 'https://www.sharesansar.com/company/nabil'},

  
    
}

#Generic File Paths (with named placeholders)
DATABASE_PATH = 'stock_data.db'
MODEL_PATH_TEMPLATE = 'models/{symbol}_model.keras'
MAIN_SCALER_PATH_TEMPLATE = 'models/{symbol}_main_scaler.pkl'
CLOSE_SCALER_PATH_TEMPLATE = 'models/{symbol}_close_scaler.pkl'
FORECAST_PLOT_TEMPLATE = 'plots/{symbol}_forecast_plot.png'
EVAL_PLOT_TEMPLATE = 'plots/{symbol}_evaluation_plot.png'

#Model Configuration
LOOK_BACK_DAYS = 14
FORECAST_HORIZON = 14
TOTAL_ROWS_TO_SCRAPE = 4000

#Scheduling Configuration
MARKET_HOLIDAYS_WEEKDAY = [4, 5] # Friday and Saturday
PUBLIC_HOLIDAYS = []