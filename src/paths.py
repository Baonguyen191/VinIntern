import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Inputs (normalized source data; parquet files are gitignored)
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'normalized')
TELEMETRY_M1_PATH = os.path.join(DATA_DIR, 'telemetry_M1.parquet')
TELEMETRY_M2_PATH = os.path.join(DATA_DIR, 'telemetry_M2.parquet')
EQUIPMENT_PATH = os.path.join(DATA_DIR, 'equipment_params.csv')
TARIFF_PATH = os.path.join(DATA_DIR, 'tariff_params.csv')

# Real outputs delivered by teammates 1 & 2 (used when USE_MOCK = False)
TEAMMATES_DIR = os.path.join(ROOT_DIR, 'data', 'teammates')
REAL_BASELINE_PATH = os.path.join(TEAMMATES_DIR, 'baseline_output.csv')
REAL_ANOMALY_PATH = os.path.join(TEAMMATES_DIR, 'anomaly_output.csv')

# Mock fixtures standing in for teammate outputs
FIXTURES_DIR = os.path.join(ROOT_DIR, 'tests', 'fixtures')
MOCK_BASELINE_PATH = os.path.join(FIXTURES_DIR, 'mock_baseline.csv')
MOCK_ANOMALY_PATH = os.path.join(FIXTURES_DIR, 'mock_anomaly.csv')

# Generated outputs (overwritten on every pipeline run)
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
FORECAST_PARQUET_PATH = os.path.join(OUTPUTS_DIR, 'forecast_test_results.parquet')
REGRESSION_COMP_PATH = os.path.join(OUTPUTS_DIR, 'regression_vs_baseline.csv')
SCENARIOS_PATH = os.path.join(OUTPUTS_DIR, 'optimization_scenarios_preliminary.csv')
FORECAST_REPORT_PATH = os.path.join(OUTPUTS_DIR, 'forecast_final_report.csv')
RANKING_CSV_PATH = os.path.join(OUTPUTS_DIR, 'recommendation_ranking.csv')
