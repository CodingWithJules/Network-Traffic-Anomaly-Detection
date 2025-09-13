# Network Anomaly Detection with Autoencoder

## Overview
This project implements an autoencoder-based anomaly detection system for network traffic data, using the CICIDS2017 dataset. The script (`anomalyDetect.py`) processes network flow data, trains an autoencoder on benign traffic, and detects anomalies (e.g., DDoS attacks) in test data. Key features include data validation with Pydantic, robust preprocessing with pandas and NumPy, and evaluation metrics (precision, recall, F1-score).

- **Datasets**:
  - `Monday-WorkingHours.pcap_ISCX.csv` (benign training data)
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (test data with DDoS attacks)
  - Download from the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) and place in the project directory.

## Recent Results
The script was successfully executed with the following results:

### Training Metrics
- **Dataset**: `Monday-WorkingHours.pcap_ISCX.csv` (benign traffic only)
- **Mean MSE**: 0.0240
- **Standard Deviation**: 0.0455
- **Max MSE**: 2.3233
- **Anomaly Threshold (90th percentile)**: 0.0613
- **Epochs**: 25 (for testing)

### Test Metrics
- **Dataset**: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (~225,000 rows, including benign and DDoS traffic)
- **Mean MSE**: 0.1618
- **Standard Deviation**: 0.4049
- **Max MSE**: 1.7575
- **Percentiles**: 50th=0.0030, 75th=0.0234, 90th=0.7131, 95th=1.4341
- **Anomaly Rate**: 18.18% of samples flagged as anomalies
- **Evaluation Metrics**:
  - **Precision**: 0.5383 (53.83% of flagged anomalies were true positives)
  - **Recall**: 0.1725 (17.25% of true anomalies detected)
  - **F1-score**: 0.2613

### Sample Predictions
Three sample inputs were tested:
- **Sample 1** (DDoS-like, `flow_packets_s=1000.0`, `flow_bytes_s=500000.0`): MSE = 2.7895, **Anomaly = True**
- **Sample 2** (Normal-like): MSE = 0.0027, **Anomaly = False**
- **Sample 3** (Normal-like): MSE = 0.0010, **Anomaly = False**

The correct classification of Sample 1 as an anomaly indicates improved preprocessing, but the low recall suggests room for optimization.

## Key Features
- **Data Validation**: Uses Pydantic to validate input data (e.g., sample inputs).
- **Preprocessing**: Handles NaNs, infinities, and outliers with quantile-based clipping (0.95 for training, 0.99 for test).
- **Autoencoder**: Trains on benign data to learn normal patterns, detecting anomalies via reconstruction error.
- **Evaluation**: Computes precision, recall, and F1-score against true labels.

## Steps Forward
The current results show a functional anomaly detection system, but the low recall (0.1725) indicates that many DDoS attacks are missed. Given the large dataset size (~225,000 rows in the test set), the following hypothesis could improve performance:

### Hypothesis: Increasing Epochs for Better Anomaly Detection
- **Rationale**: The autoencoder was trained for only 25 epochs to facilitate testing, which may lead to underfitting, especially with a large dataset of hundreds of thousands of rows. Increasing the number of epochs (e.g., to 50 or 100) could allow the autoencoder to better learn the patterns of benign traffic, improving its ability to detect anomalies by producing higher reconstruction errors for DDoS samples.
- **Proposed Change**:
  ```python
  autoencoder.fit(train_data, train_data, epochs=50, batch_size=256, validation_split=0.1, verbose=1)
  ```
- **Expected Impact**: Higher epochs should reduce the training MSE and increase the separation between benign and anomalous MSE distributions, potentially improving recall and F1-score.

## Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, feature requests, or optimizations.

## License
This project is licensed under the MIT License.