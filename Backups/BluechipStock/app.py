import os
import atexit
from datetime import datetime
from flask import Flask, send_file, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from main import main_orchestrator
import config
import database

app = Flask(__name__)
CORS(app)

def scheduled_job():
    """The function the scheduler will run."""
    print(f"\n--- SCHEDULER: Kicking off daily pipeline run at {datetime.now()} ---")
    main_orchestrator()
    print("--- SCHEDULER: Daily run finished. ---")

scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Kathmandu')
# Runs daily at 5:00 PM Nepal time
scheduler.add_job(scheduled_job, 'cron', hour=0, minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

@app.route('/forecast/<symbol>')
def get_forecast(symbol):
    """API endpoint to serve the forecast image for a specific stock."""
    image_path = config.FORECAST_PLOT_TEMPLATE.format(symbol=symbol.upper())
    if not os.path.exists(image_path):
        return jsonify({"error": f"Forecast for {symbol.upper()} not found."}), 404
    return send_file(image_path, mimetype='image/png')

@app.route('/evaluation/<symbol>')
def get_evaluation(symbol):
    """API endpoint to serve the evaluation image for a specific stock."""
    image_path = config.EVAL_PLOT_TEMPLATE.format(symbol=symbol.upper())
    if not os.path.exists(image_path):
        return jsonify({"error": f"Evaluation for {symbol.upper()} not found."}), 404
    return send_file(image_path, mimetype='image/png')

@app.route('/')
def index():
    return "Hello! The BluechipStocks forecast server is running."

if __name__ == '__main__':
    print("--- Initializing Database ---")
    for symbol in config.STOCKS.keys():
        database.init_db(symbol)
    
    # Run the pipeline once on startup for any stock that doesn't have a forecast plot yet
    for symbol in config.STOCKS.keys():
        plot_path = config.FORECAST_PLOT_TEMPLATE.format(symbol=symbol)
        if not os.path.exists(plot_path):
            print(f"--- STARTUP: No forecast plot found for {symbol}. Running pipeline once to generate it... ---")
            from main import run_pipeline_for_stock
            run_pipeline_for_stock(symbol, config.STOCKS[symbol])
            print(f"--- STARTUP: Initial pipeline run for {symbol} complete. ---")
    
    print("\n--- Starting Flask Server ---")
    app.run(debug=False)