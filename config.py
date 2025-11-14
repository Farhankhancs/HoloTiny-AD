
# Random seed for reproducibility
RANDOM_STATE = 42

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    'missing_value_threshold': 0.5,  # Remove columns with >50% missing values
    'high_cardinality_threshold': 0.9,  # Remove columns with >90% unique values
    'rolling_window_size': 5,  # For temporal feature extraction
    'holographic_feature_prefixes': ['temporal_', 'spatial_', 'behavioral_', 'correlation_']
}

# Features to explicitly remove (if known in advance)
EXPLICITLY_REMOVED_FEATURES = [
    'flow_id', 'timestamp', 'session_id', 'device_id',
    'src_mac', 'dst_mac', 'user_agent'
]

# Feature encoding configuration
ENCODING_CONFIG = {
    'unknown_category_handling': 'encode_as_unknown',
    'max_categories_for_one_hot': 10
}

# Holographic feature extraction parameters
HOLOGRAPHIC_FEATURE_CONFIG = {
    'temporal_features': ['rolling_mean', 'rolling_std', 'trend'],
    'spatial_features': ['centrality', 'distance'],
    'behavioral_features': ['state_change', 'usage_intensity', 'anomaly_score'],
    'correlation_features': ['interaction', 'ratio', 'summary_stats']
}