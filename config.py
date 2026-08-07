# =============================================================================
# BATTERY PARAMETERS
# =============================================================================
BATTERY_CONFIG = {
    'P': 30, # in MW
    'E': 100, # in MWh
    'eta_ch': 0.92,
    'eta_dis': 0.92,
    'C': 2.5, # EUR/MWh
}

# =============================================================================
# DATA PARAMETERS
# =============================================================================
DATA_CONFIG = {
    'batch_size': 35,
    'seq_len': 168,  # Lookback: 168 hours (Exactly 1 week of historical data)
    'label_len': 48,  # Overlap section used internally by specific Transformer variants
    'pred_len': 24,  # Horizon: 24 hours (Predict next day's electricity prices)
    'features': 'MS',
    'target': 'OT',
    'embed': 'timeF',
    'freq': 'h',
    'root_path': '../timexer/dataset/EPF/',
    'data_path': 'DE.csv',
    'num_workers': 4,
    'augmentation_ratio': 0,
}