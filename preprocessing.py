import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HolographicDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for holographic feature extraction and data preparation
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scalers = {}
        self.feature_info = {}
        self.removed_features = []

    def preprocess_data(self, df, target_column=None):
        """
        Main preprocessing function that handles feature removal, encoding, and normalization

        Parameters:
        df: pandas DataFrame - input dataset
        target_column: str - name of target column (if None, last column is used)

        Returns:
        preprocessed_df: pandas DataFrame - cleaned and processed data
        preprocessing_report: dict - detailed report of preprocessing steps
        """

        logger.info(f"Starting preprocessing pipeline")
        logger.info(f"Original dataset shape: {df.shape}")

        # Create a copy to avoid modifying original data
        df_processed = df.copy()

        # Identify target column
        if target_column is None:
            target_column = df_processed.columns[-1]
            logger.info(f"Using last column as target: {target_column}")

        # Step 1: Remove non-numeric identifiers and metadata columns
        df_processed, removed_features = self._remove_metadata_columns(df_processed, target_column)
        self.removed_features = removed_features

        # Step 2: Handle missing values
        df_processed = self._handle_missing_values(df_processed)

        # Step 3: Encode categorical variables
        df_processed = self._encode_categorical_features(df_processed, target_column)

        # Step 4: Extract holographic features
        df_processed = self._extract_holographic_features(df_processed, target_column)

        # Step 5: Feature scaling and normalization
        df_processed = self._scale_features(df_processed, target_column)

        # Generate preprocessing report
        preprocessing_report = self._generate_preprocessing_report(df, df_processed, target_column)

        logger.info("Preprocessing pipeline completed successfully")
        return df_processed, preprocessing_report

    def _remove_metadata_columns(self, df, target_column):
        """
        Remove non-numeric identifiers and metadata columns that are not suitable for ML
        """
        logger.info("Step 1: Removing metadata columns")

        features_to_remove = []
        features_to_keep = []

        for column in df.columns:
            if column == target_column:
                features_to_keep.append(column)
                continue

            # Remove columns with high cardinality (like IDs) or mostly unique values
            if df[column].nunique() / len(df) > 0.9:
                features_to_remove.append(column)
                logger.info(f"  Removing high-cardinality column: {column} ({df[column].nunique()} unique values)")
            # Remove columns with excessive missing values (>50%)
            elif df[column].isnull().sum() / len(df) > 0.5:
                features_to_remove.append(column)
                logger.info(f"  Removing column with excessive missing values: {column}")
            # Remove constant columns
            elif df[column].nunique() <= 1:
                features_to_remove.append(column)
                logger.info(f"  Removing constant column: {column}")
            else:
                features_to_keep.append(column)

        df_processed = df[features_to_keep]
        logger.info(f"  Removed {len(features_to_remove)} metadata columns")
        logger.info(f"  Remaining {len(features_to_keep)} columns")

        return df_processed, features_to_remove

    def _handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        """
        logger.info("Step 2: Handling missing values")

        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"  Found {missing_before} missing values")

            # For numeric columns, fill with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
                    logger.info(f"  Filled missing values in {col} with median")

            # For categorical columns, fill with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                    logger.info(f"  Filled missing values in {col} with mode")

        missing_after = df.isnull().sum().sum()
        logger.info(f"  Missing values after handling: {missing_after}")

        return df

    def _encode_categorical_features(self, df, target_column):
        """
        Encode categorical variables using Label Encoding
        """
        logger.info("Step 3: Encoding categorical features")

        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]

        self.label_encoders = {}

        for col in categorical_columns:
            logger.info(f"  Encoding categorical column: {col}")
            self.label_encoders[col] = LabelEncoder()

            # Handle unseen categories by fitting on current data
            df[col] = df[col].astype(str)
            self.label_encoders[col].fit(df[col])
            df[col] = self.label_encoders[col].transform(df[col])

            logger.info(f"    Original categories: {len(self.label_encoders[col].classes_)}")
            logger.info(f"    Encoded values range: {df[col].min()} to {df[col].max()}")

        logger.info(f"  Encoded {len(categorical_columns)} categorical columns")
        return df

    def _extract_holographic_features(self, df, target_column):
       
        logger.info("Step 4: Extracting holographic features")

        feature_columns = [col for col in df.columns if col != target_column]

        # F_temporal: Temporal patterns using rolling statistics
        temporal_features = self._extract_temporal_features(df, feature_columns)

        # F_spatial: Network topology and spatial relationships
        spatial_features = self._extract_spatial_features(df, feature_columns)

        # F_behavioral: Device state transitions and behavioral patterns
        behavioral_features = self._extract_behavioral_features(df, feature_columns)

        # F_correlation: Cross-feature relationships
        correlation_features = self._extract_correlation_features(df, feature_columns)

        # Combine all holographic features
        holographic_features = {}
        holographic_features.update(temporal_features)
        holographic_features.update(spatial_features)
        holographic_features.update(behavioral_features)
        holographic_features.update(correlation_features)

        # Add holographic features to dataframe
        for feature_name, feature_values in holographic_features.items():
            if len(feature_values) == len(df):
                df[feature_name] = feature_values
                logger.info(f"  Added holographic feature: {feature_name}")

        logger.info(f"  Extracted {len(holographic_features)} holographic features")
        return df

    def _extract_temporal_features(self, df, feature_columns):
        """Extract temporal patterns using rolling statistics"""
        temporal_features = {}

        for col in feature_columns[:3]:  # Use first 3 features for temporal analysis
            if df[col].dtype in [np.int64, np.float64]:
                # Rolling mean and standard deviation
                temporal_features[f'temporal_rolling_mean_{col}'] = df[col].rolling(window=5, min_periods=1).mean()
                temporal_features[f'temporal_rolling_std_{col}'] = df[col].rolling(window=5, min_periods=1).std()

                # Trend features
                temporal_features[f'temporal_trend_{col}'] = df[col].diff().rolling(window=3, min_periods=1).mean()

        return temporal_features

    def _extract_spatial_features(self, df, feature_columns):
        """Extract spatial topology features"""
        spatial_features = {}

        if len(feature_columns) >= 2:
            # Network centrality-like features
            for i, col1 in enumerate(feature_columns[:2]):
                spatial_features[f'spatial_centrality_{col1}'] = (df[col1] - df[col1].mean()) / df[col1].std()

            # Distance-based features
            if len(feature_columns) >= 3:
                col1, col2, col3 = feature_columns[:3]
                spatial_features['spatial_feature_distance'] = np.sqrt(
                    df[col1] ** 2 + df[col2] ** 2 + df[col3] ** 2
                )

        return spatial_features

    def _extract_behavioral_features(self, df, feature_columns):
        """Extract behavioral state transition features"""
        behavioral_features = {}

        for col in feature_columns[:3]:  # Use first 3 features for behavioral analysis
            if df[col].dtype in [np.int64, np.float64]:
                # State transition indicators
                behavioral_features[f'behavioral_state_change_{col}'] = df[col].diff().abs()

                # Usage patterns
                behavioral_features[f'behavioral_usage_intensity_{col}'] = df[col] / df[col].max()

                # Anomaly likelihood based on z-score
                z_score = (df[col] - df[col].mean()) / df[col].std()
                behavioral_features[f'behavioral_anomaly_score_{col}'] = z_score.abs()

        return behavioral_features

    def _extract_correlation_features(self, df, feature_columns):
        """Extract cross-feature correlation patterns"""
        correlation_features = {}

        if len(feature_columns) >= 2:
            # Feature interaction terms
            for i in range(min(2, len(feature_columns))):
                for j in range(i + 1, min(3, len(feature_columns))):
                    col1, col2 = feature_columns[i], feature_columns[j]
                    correlation_features[f'correlation_interaction_{col1}_{col2}'] = df[col1] * df[col2]
                    correlation_features[f'correlation_ratio_{col1}_{col2}'] = df[col1] / (df[col2] + 1e-8)

            # Multi-feature correlation summary
            if len(feature_columns) >= 3:
                correlation_features['correlation_feature_sum'] = df[feature_columns[:3]].sum(axis=1)
                correlation_features['correlation_feature_variance'] = df[feature_columns[:3]].var(axis=1)

        return correlation_features

    def _scale_features(self, df, target_column):
        """
        Apply feature scaling using StandardScaler
        """
        logger.info("Step 5: Scaling features")

        feature_columns = [col for col in df.columns if col != target_column]

        self.scalers['feature_scaler'] = StandardScaler()
        df[feature_columns] = self.scalers['feature_scaler'].fit_transform(df[feature_columns])

        logger.info(f"  Scaled {len(feature_columns)} feature columns")
        return df

    def _generate_preprocessing_report(self, original_df, processed_df, target_column):
        """
        Generate detailed preprocessing report
        """
        report = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'removed_features': self.removed_features,
            'removed_features_count': len(self.removed_features),
            'categorical_columns_encoded': list(self.label_encoders.keys()),
            'label_encoders_used': len(self.label_encoders),
            'scalers_applied': len(self.scalers),
            'feature_columns_final': [col for col in processed_df.columns if col != target_column],
            'target_column': target_column,
            'holographic_features_added': [col for col in processed_df.columns
                                           if any(prefix in col for prefix in ['temporal_', 'spatial_',
                                                                               'behavioral_', 'correlation_'])]
        }

        logger.info("Preprocessing Report:")
        logger.info(f"  Original dataset: {report['original_shape']}")
        logger.info(f"  Processed dataset: {report['processed_shape']}")
        logger.info(f"  Features removed: {report['removed_features_count']}")
        logger.info(f"  Categorical columns encoded: {report['label_encoders_used']}")
        logger.info(f"  Holographic features added: {len(report['holographic_features_added'])}")

        return report


# Example usage function
def demonstrate_preprocessing():
    """
    Demonstrate the preprocessing pipeline with sample data
    """
    # Create sample data similar to IoT network traffic
    sample_data = {
        'src_ip': ['192.168.1.1', '192.168.1.2', '192.168.1.1', '192.168.1.3'] * 25,
        'dst_ip': ['10.0.0.1', '10.0.0.2', '10.0.0.1', '10.0.0.3'] * 25,
        'protocol': ['TCP', 'UDP', 'TCP', 'ICMP'] * 25,
        'packet_count': np.random.randint(1, 1000, 100),
        'byte_count': np.random.randint(64, 1500, 100),
        'duration': np.random.uniform(0.1, 60.0, 100),
        'attack_type': ['Normal', 'DDoS', 'Normal', 'PortScan'] * 25
    }

    df = pd.DataFrame(sample_data)
    print("Original sample data:")
    print(df.head())
    print(f"Shape: {df.shape}")

    # Initialize preprocessor
    preprocessor = HolographicDataPreprocessor(random_state=42)

    # Apply preprocessing
    processed_df, report = preprocessor.preprocess_data(df, target_column='attack_type')

    print("\nProcessed data:")
    print(processed_df.head())
    print(f"Shape: {processed_df.shape}")

    print("\nPreprocessing Report Summary:")
    print(f"Removed {report['removed_features_count']} features")
    print(f"Encoded {report['label_encoders_used']} categorical columns")
    print(f"Added {len(report['holographic_features_added'])} holographic features")

    return processed_df, report


if __name__ == "__main__":
    # Run demonstration
    processed_data, preprocessing_report = demonstrate_preprocessing()