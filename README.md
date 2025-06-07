# Multimodal_duplicate_detection
Multimodal AI framework to detect duplicates in the absence of PII identifiers
# Multimodal Duplicate Detection in CRM Data

This repository accompanies our research paper on AI-powered deduplication using semantic, behavioral, and device-level signals.

## 📂 Files

- `src/customer_duplicate_detector_2.py`: Python script implementing the deduplication pipeline
- `code/Simulated_CRM_Dataset.csv`: Input CRM dataset (synthetic, 200 records)
- `code/Simulated_CRM_Dataset_duplicates.csv`: Output duplicate predictions with cluster and similarity scores

## ⚙️ Requirements

- Python 3.8+
- `transformers`, `torch`, `scikit-learn`, `pandas`, `numpy`

Install with:

```bash
pip install -r requirements.txt

