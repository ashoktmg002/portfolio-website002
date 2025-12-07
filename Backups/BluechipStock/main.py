import os
from datetime import datetime
import pytz
import config
import scraper
import model_pipeline
import database

def is_market_open_today():
    nepal_tz = pytz.timezone('Asia/Kathmandu')
    today = datetime.now(nepal_tz)
    if today.weekday() in config.MARKET_HOLIDAYS_WEEKDAY:
        print(f"Market is closed today (Weekend). No action needed.")
        return False
    if today.strftime('%Y-%m-%d') in config.PUBLIC_HOLIDAYS:
        print(f"Market is closed today (Public Holiday). No action needed.")
        return False
    return True

def run_pipeline_for_stock(symbol, stock_info):
    """Runs the entire pipeline for a single stock."""
    print(f"\n{'='*15} Processing Stock: {symbol.upper()} {'='*15}")
    
    was_updated = scraper.scrape_and_save_data(symbol, stock_info)
    
    model_path = config.MODEL_PATH_TEMPLATE.format(symbol=symbol)

    if was_updated or not os.path.exists(model_path):
        if not os.path.exists(model_path):
            print(f"No pre-trained model found for {symbol}. Must train a new one.")
        else:
            print(f"New data found for {symbol}. Proceeding with full training pipeline...")
        model_pipeline.run_prediction_pipeline(symbol)
    else:
        print(f"Dataset for {symbol} is up-to-date. Generating forecast from saved model...")
        model_pipeline.generate_forecast_from_saved_model(symbol)

def main_orchestrator():
    """Loops through all stocks in the config and runs the pipeline for each."""
    if is_market_open_today():
        print("Market is open. Running the pipeline for all configured stocks...")
        for symbol, info in config.STOCKS.items():
            try:
                run_pipeline_for_stock(symbol, info)
            except Exception as e:
                print(f"!!!!!! A CRITICAL ERROR occurred while processing {symbol}. Error: {e} !!!!!!")
    else:
        print("Market is closed. No actions taken.")

if __name__ == "__main__":
    print("--- Starting Daily Stock Prediction Pipeline ---")
    
    # Initialize a separate table for each stock
    for symbol in config.STOCKS.keys():
        database.init_db(symbol)
    
    main_orchestrator()
    
    print("\n--- Pipeline has completed its run for today. ---")