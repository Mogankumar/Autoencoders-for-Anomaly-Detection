# EC2 CPU Anomaly Detection — Autoencoder

Unsupervised anomaly detection on AWS EC2 CPU utilization using a deep autoencoder built with PyTorch. Achieves AUROC 0.9998 with zero false positives on the Numenta Anomaly Benchmark (NAB) dataset.

---

## Requirements

- Python 3.8+
- pip

---

## Setup

1. **Clone the repo**
```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Add the dataset**
   
   Place `ec2_cpu_utilization_ac20cd.csv` in the root directory.  
   Dataset source: [Numenta Anomaly Benchmark](https://github.com/numenta/NAB)

---

## Project Structure

```
├── ec2_cpu_utilization_ac20cd.csv   # dataset
├── anomaly_detection.ipynb                # main notebook
├── autoencoder_anomaly.pt         
├── requirements.txt
└── README.md
```

---

## Usage

Open and run the notebook end to end:

```bash
jupyter notebook autoencoder.ipynb
```

Or run as a script if exported:

```bash
python autoencoder.py
```

The notebook is structured in sequential steps:

| Step | Description |
|------|-------------|
| 1 | Imports and reproducibility |
| 2 | EDA and visualisation |
| 3 | Preprocessing and sliding windows |
| 4 | Autoencoder architecture |
| 5 | Training with early stopping |
| 6 | Reconstruction error computation |
| 7 | Threshold calibration |
| 8 | Evaluation — AUROC, F1, confusion matrix |
| 9 | Full results visualisation |

---

## Results

| Metric | Score |
|--------|-------|
| AUROC | 0.9998 |
| AUPRC | 0.9986 |
| F1 | 0.9857 |
| False Positive Rate | 0.0000 |
| False Negative Rate | 0.0302 |

---

## requirements.txt

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

---

## How it works

The autoencoder is trained **only on normal CPU behaviour**. At inference time, normal windows are reconstructed accurately (low MSE). Anomalous windows, patterns the model has never seen, cannot be reconstructed well, producing a high reconstruction error that triggers the anomaly flag.

---

## License

MIT
