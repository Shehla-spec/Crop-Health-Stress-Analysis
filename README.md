# 🌾 Crop-Stress-Numerical-Validation

**Large-Scale Biophysical Analysis of Abiotic Stress in Cereal Crops**

This repository provides a high-throughput analytical framework for validating plant health under environmental stressors. Utilizing a dataset of **212,000+ records**, this project identifies physiological tipping points for **Wheat, Maize, and Rice** through multivariate numerical modeling.

---

## 🔬 Dataset Overview

While initiatives like PlantVillage focus on visual symptoms, this repository targets the **biophysical precursors** of crop failure.
* **Scale:** 212,000+ Observations
* **Size:** 73.5 MB (CSV)
* **Parameters:** $N, P, K$ (Soil Chemistry), Temperature, Humidity, pH, and Rainfall.

---

## ⚙️ Installation

To set up the research environment, ensure you have Python 3.8+ installed:

git clone https://github.com/Shehla-spec/Crop-Health-Stress-Analysis
cd Crop-Health-Stress-Analysis
pip install pandas scikit-learn seaborn matplotlib

# 📂 Analytical Pipeline

The core logic utilizes a Random Forest Regressor to extract feature importance, identifying which abiotic stressors dictate yield decline.

from src.analysis import validate_stress

# Execute validation on the 212k record dataset
model_results = validate_stress("data/crop_health_data.csv")


# 📊 Data Access (73.5 MB)

Note: Due to GitHub's browser upload limitations, the raw CSV is hosted via Kaggle for stability and high-speed access.

Dataset Link: https://www.kaggle.com/datasets/datasetengineer/crop-health-and-environmental-stress-dataset/discussion?sort=hotness

# 👨‍🔬 Author Information

Academic Researcher | Focus: Botany • Plant Physiology • Climate Resilience

Education: B.Sc. (Hons) Botany, University of Peshawar
