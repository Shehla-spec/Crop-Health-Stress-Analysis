# 🌾 Crop-Stress-Numerical-Validation
**An Open Access Repository for Large-Scale Abiotic Stress Modeling**

This repository provides a high-throughput analytical framework for validating plant health under environmental stressors. Inspired by the **PlantVillage** initiative, this project shifts the focus from image-based diagnosis to **multivariate numerical validation**, utilizing a dataset of **212,000+ records** across Wheat, Maize, and Rice.

---

## 🔬 Dataset Overview
While initiatives like PlantVillage focus on visual symptoms, this dataset targets the **physiological precursors** of crop failure.
* **Records:** 212,000+ observations.
* **Parameters:** $N, P, K$ (Soil Chemistry), Temperature, Humidity, pH, and Rainfall.
* **Objective:** Automated yield prediction and stress-bottleneck identification.

---

## 🚀 Usage & Implementation

### Prerequisites
* **Python 3.8+** (Optimized for modern data pipelines)
* **Libraries:** `pandas`, `scikit-learn`, `seaborn`, `matplotlib`
* Analytical Pipeline
The core logic utilizes a Random Forest Regressor to extract feature importance, identifying which abiotic stressors dictate yield decline.

### Installation
```bash
git clone [https://github.com/YOUR_USERNAME/Crop-Health-Analysis](https://github.com/YOUR_USERNAME/Crop-Health-Analysis)
cd Crop-Health-Analysis
pip install -r requirements.txt
Analytical Pipeline
The core logic utilizes a Random Forest Regressor to extract feature importance, identifying which abiotic stressors dictate yield decline.

Python
from src.analysis import validate_stress
# Run the validation on the 212k record dataset
model_results = validate_stress("data/crop_health_data.csv")
📂 Repository Structure
.
├── src/                  # Core Analytical Logic
│   └── analysis.py       # ML Pipeline (Random Forest)
├── results/              # Biophysical Validation Plots
│   ├── heatmap.png       # Correlation Matrix
│   └── importance.png    # Stressor Rank Analysis
├── data_info/            # Metadata and Data Dictionary
├── LICENSE               # MIT Open Access
└── README.md             # Research Documentation
📊 Data Access (73.5 MB)
Due to GitHub's browser upload limitations, the raw CSV (212k records) is hosted via Kaggle for stability and high-speed access.
## Author
**Shehla Sajjad**
BSc (Hons) Botany |Plant Physiology Specialization 

