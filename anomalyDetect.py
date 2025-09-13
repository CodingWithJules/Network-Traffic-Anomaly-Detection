import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Define Pydantic model for data validation
class NetworkLog(BaseModel):
    flow_duration: float = Field(..., ge=0, description="Duration of the flow in microseconds")
    total_fwd_packets: int = Field(..., ge=0, description="Total packets in forward direction")
    flow_iat_mean: float = Field(..., ge=0, description="Mean inter-arrival time of flow")
    destination_port: int = Field(..., ge=0, le=65535, description="Destination port number")
    psh_flag_count: int = Field(..., ge=0, le=1, description="Count of PSH flags (0 or 1)")
    flow_packets_s: float = Field(..., ge=0, description="Flow packets per second")
    flow_bytes_s: float = Field(..., ge=0, description="Flow bytes per second")

# Step 2: Load and preprocess dataset
def load_and_preprocess(
    file_path: str,
    features: List[str],
    label_column: Optional[str] = None,
    is_training: bool = False,
    scale: bool = True
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[np.ndarray]]:
    # Read CSV once with all columns
    df = pd.read_csv(file_path, low_memory=False)
    logging.info(f"Columns in {file_path}: {df.columns.tolist()}")
    logging.info(f"Initial shape of {file_path}: {df.shape}")

    # Check for missing columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")

    # Select features and labels together
    columns_to_select = features.copy()
    if label_column and label_column in df.columns:
        columns_to_select.append(label_column)
    elif label_column:
        raise KeyError(f"Label column '{label_column}' not found.")

    df_selected = df[columns_to_select]
    logging.info(f"Shape after selecting columns: {df_selected.shape}")

    # Drop NaNs across all selected columns
    df_selected = df_selected.dropna()
    logging.info(f"Shape after dropping NaNs: {df_selected.shape}")

    # Extract features and labels before filtering infinites
    df_features = df_selected[features]
    labels = None
    if label_column:
        labels = df_selected[label_column].values
        logging.info(f"Label distribution in {file_path}:")
        logging.info(df_selected[label_column].value_counts())

    # Filter rows with infinite values in features only
    mask_finite = np.isfinite(df_features.select_dtypes(include=[np.number])).all(axis=1)
    df_features = df_features[mask_finite]
    logging.info(f"df_features shape after dropping infinities: {df_features.shape}")

    # Align labels with filtered features
    if label_column:
        labels = labels[mask_finite]
        logging.info(f"Labels shape after aligning with features: {labels.shape}")

    # Apply benign filter for training data
    if is_training and label_column:
        mask_benign = (labels == 'BENIGN')
        df_features = df_features[mask_benign]
        labels = labels[mask_benign]
        logging.info(f"Shape after filtering benign data: {df_features.shape}")
        logging.info(f"Labels shape after benign filter: {labels.shape}")

    # Cap large values in df_features
    for col in df_features.columns:
        max_val = df_features[col].quantile(0.95)  # Stricter clipping for outliers
        df_features[col] = df_features[col].clip(upper=max_val)
    logging.info(f"df_features shape after clipping: {df_features.shape}")

    # Convert features to numpy and optionally scale
    data = df_features.values
    scaler = None
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    logging.info(f"Final shape of processed data: {data.shape}")
    if labels is not None:
        logging.info(f"Final labels shape: {labels.shape}")

    return data, scaler, labels

# Step 3: Build autoencoder
def build_autoencoder(input_dim: int):
    encoder = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu')
    ])
    decoder = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(16,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Step 4: Validate inputs with Pydantic
def validate_logs(inputs: List[dict], expected_features: List[str]) -> Optional[np.ndarray]:
    valid_data = []
    for item in inputs:
        try:
            log = NetworkLog(**item)
            # Ensure order matches expected_features
            data_row = []
            for feature in expected_features:
                if feature == ' Flow Duration':
                    data_row.append(log.flow_duration)
                elif feature == ' Total Fwd Packets':
                    data_row.append(log.total_fwd_packets)
                elif feature == ' Flow IAT Mean':
                    data_row.append(log.flow_iat_mean)
                elif feature == ' Destination Port':
                    data_row.append(log.destination_port)
                elif feature == ' PSH Flag Count':
                    data_row.append(log.psh_flag_count)
                elif feature == ' Flow Packets/s':
                    data_row.append(log.flow_packets_s)
                elif feature == ' Flow Bytes/s':
                    data_row.append(log.flow_bytes_s)
            valid_data.append(data_row)
        except ValidationError as e:
            logging.warning(f"Invalid log entry {item}: {e}")
    return np.array(valid_data) if valid_data else None

# Step 5: Detect anomalies and evaluate
def detect_anomalies(
    model,
    data: np.ndarray,
    scaler: StandardScaler,
    threshold: float,
    true_labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    data_scaled = scaler.transform(data) if data.ndim == 2 else scaler.transform(data.reshape(1, -1))
    reconstructions = model.predict(data_scaled, verbose=0)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    anomalies = mse > threshold

    metrics = None
    if true_labels is not None:
        binary_true = (true_labels != 'BENIGN').astype(int)
        metrics = {
            'precision': precision_score(binary_true, anomalies),
            'recall': recall_score(binary_true, anomalies),
            'f1': f1_score(binary_true, anomalies)
        }
        logging.info(f"Evaluation metrics - Precision: {metrics['precision']:.4f}, "
                     f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    return anomalies, mse, metrics

# Step 6: Visualize MSE distribution
def plot_mse_histogram(train_mse, test_mse, threshold, save_path='mse_histogram.png'):
    plt.figure(figsize=(10, 6))
    plt.hist(train_mse, bins=50, alpha=0.5, label='Training (Normal)', color='blue')
    plt.hist(test_mse, bins=50, alpha=0.5, label='Test (Mixed)', color='orange')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold (90th: {threshold:.4f})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('MSE Distribution for Anomaly Detection')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Main execution
if __name__ == "__main__":
    try:
        # Initial feature list based on working columns + likely additions
        features = [
            ' Flow Duration', ' Total Fwd Packets', ' Flow IAT Mean', ' Destination Port',
            ' PSH Flag Count', ' Flow Packets/s', ' Flow Bytes/s'  # Adjusted names
        ]
        label_col = ' Label'

        # Debug: Load and print columns to confirm available names
        df_train = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv', low_memory=False)
        logging.info(f"Available columns in training data: {df_train.columns.tolist()}")
        df_test = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', low_memory=False)
        logging.info(f"Available columns in test data: {df_test.columns.tolist()}")

        # Filter features to only those present in the dataset
        available_features = [col for col in features if col in df_train.columns]
        logging.info(f"Using available features: {available_features}")

        # Load training data (benign only, scaled)
        train_data, scaler, _ = load_and_preprocess(
            'Monday-WorkingHours.pcap_ISCX.csv', available_features, label_col, is_training=True, scale=True
        )

        # Build and train autoencoder with more epochs
        input_dim = train_data.shape[1]
        autoencoder = build_autoencoder(input_dim)
        autoencoder.fit(train_data, train_data, epochs=25, batch_size=256, validation_split=0.1, verbose=1)

        # Compute threshold (90th percentile)
        train_recon = autoencoder.predict(train_data, verbose=0)
        train_mse = np.mean(np.power(train_data - train_recon, 2), axis=1)
        threshold = np.percentile(train_mse, 85)
        logging.info(f"Training MSE - Mean: {np.mean(train_mse):.4f}, Std: {np.std(train_mse):.4f}")
        logging.info(f"Training MSE - Max: {np.max(train_mse):.4f}")
        logging.info(f"Anomaly Detection Threshold (90th percentile): {threshold:.4f}")

        # Load test data (raw, with labels)
        test_data, _, test_labels = load_and_preprocess(
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', available_features, label_col, scale=False
        )

        # Detect anomalies
        anomalies, test_mse, metrics = detect_anomalies(autoencoder, test_data, scaler, threshold, test_labels)
        anomaly_rate = np.mean(anomalies) * 100
        logging.info(f"Test MSE - Mean: {np.mean(test_mse):.4f}, Std: {np.std(test_mse):.4f}")
        logging.info(f"Test MSE - Max: {np.max(test_mse):.4f}")
        logging.info(f"Test MSE - Percentiles: 50th={np.percentile(test_mse, 50):.4f}, "
                     f"75th={np.percentile(test_mse, 75):.4f}, 90th={np.percentile(test_mse, 90):.4f}, "
                     f"95th={np.percentile(test_mse, 95):.4f}")
        logging.info(f"Detected {anomaly_rate:.2f}% anomalies in test data (90th percentile).")

        # Visualize MSE
        plot_mse_histogram(train_mse, test_mse, threshold)
        logging.info("MSE histogram saved as 'mse_histogram.png'")

        # Example: Validate and detect on sample inputs
        sample_inputs = [
            {
                "flow_duration": 1000000.0, "total_fwd_packets": 50, "flow_iat_mean": 20000.0,
                "destination_port": 80, "psh_flag_count": 1, "flow_packets_s": 100.0, "flow_bytes_s": 500000.0
            },  # DDoS-like
            {
                "flow_duration": 500.0, "total_fwd_packets": 3, "flow_iat_mean": 100.0,
                "destination_port": 8080, "psh_flag_count": 0, "flow_packets_s": 1.0, "flow_bytes_s": 50.0
            },  # Fixed negative flow_duration
            {
                "flow_duration": 2000.0, "total_fwd_packets": 10, "flow_iat_mean": 200.0,
                "destination_port": 443, "psh_flag_count": 0, "flow_packets_s": 5.0, "flow_bytes_s": 100.0
            }  # Normal
        ]
        # Validate and detect on sample inputs
        valid_samples = validate_logs(sample_inputs, available_features)
        if valid_samples is not None and len(valid_samples) > 0:
            sample_anomalies, sample_mse, _ = detect_anomalies(autoencoder, valid_samples, scaler, threshold)
            logging.info("\nSample Predictions (90th percentile threshold):")
            for i, is_anomaly in enumerate(sample_anomalies):
                logging.info(f"Sample {i + 1}: MSE = {sample_mse[i]:.4f}, Anomaly = {is_anomaly}")

    except KeyError as e:
        logging.error(f"Error: {e}. Please check column names in the CSV files.")
    except FileNotFoundError:
        logging.error(
            "Error: CSV file not found. Ensure 'Monday-WorkingHours.pcap_ISCX.csv' and "
            "'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' are in the directory."
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}")