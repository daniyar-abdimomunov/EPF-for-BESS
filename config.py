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
# =============================================================================
# MODEL PARAMETERS
# =============================================================================
MODEL_CONFIG = {
    **DATA_CONFIG,
    'inverse': True,
    'output_attention': False,
    'loss': 'mae',
    'task_name': 'long_term_forecast',  # Primary mode for sequential prediction
    'use_norm': 1, # Boolean value for normalization
    'patch_len': 24,  # Patch size: 24 (Matches the 24h daily price cycle), default = 16
    'd_model': 512,  # d_model=64,  # Dimension size for core embeddings, default=512
    'dropout': 0.1,  # Dropout rate to combat overfitting
    'factor': 1,  # ProbSparse attention probing factor, default = 1
    'n_heads': 8,  # Attention heads, default = 8
    'd_ff': 2048,  #d_ff=128,  # Dimension size of feed-forward layers, default = 2048
    'activation':'gelu',  # Activation function mapping
    'e_layers': 2,  # Number of encoder processing layers
    'enc_in': 3, # Number of input variables (e.g., Target Price + 5 Exogenous features), default=7
    'learning_rate': 1e-4,
}

# =============================================================================
# COV-E PENALISED MODEL PARAMETERS
# =============================================================================
COV_E_CONFIG = {
    'penalty': 'cov-e',
    'penalty_lambda': 0.001,
}

# =============================================================================
# CORR-F PENALISED MODEL PARAMETERS
# =============================================================================
CORR_F_CONFIG = {
    'penalty': 'corr-f',
    'penalty_lambda': 1,
}

# =============================================================================
# SPOPLUS PENALISED MODEL PARAMETERS
# =============================================================================
SPOPLUS_CONFIG = {
    'penalty': 'spo+',
    'penalty_lambda': 0.0001,
}