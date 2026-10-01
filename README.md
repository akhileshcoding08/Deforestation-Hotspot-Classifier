# 🌳 Deforestation Hotspot Classifier

A machine learning project that predicts **deforestation hotspots** using a **Random Forest Classifier**, built on a geospatial and environmental dataset of 15,000 forest-region observations. The project covers the complete pipeline consists of exploratory data analysis, preprocessing, model training, hyperparameter tuning, and performance evaluation.

---

## 📌 Project Overview

Deforestation is one of the leading drivers of climate change and biodiversity loss. This project uses environmental, geographic, and human-activity indicators (forest loss, NDVI change, fire incidents, proximity to roads/settlements, population density, etc.) to classify whether a given region is a **deforestation hotspot (`is_hotspot = 1`)** or not.

The model is trained using a **Random Forest Classifier**, tuned using **RandomizedSearchCV** followed by **GridSearchCV**, and evaluated using standard classification metrics plus residual and ROC analysis.

---

## 📂 Dataset

**File:** `deforestation_hotspot_dataset.csv`
**Shape:** 15,000 rows × 21 columns
**Target variable:** `is_hotspot` (binary: 1 = hotspot, 0 = non-hotspot)

| Column | Description |
|---|---|
| `region` | Geographic region (e.g., Amazon Basin, Congo Basin) |
| `biome` | Biome type (e.g., Tropical Rainforest, Tropical Dry Forest) |
| `latitude`, `longitude` | Coordinates of the observation |
| `year` | Year of observation |
| `primary_driver` | Main driver of deforestation (agriculture, logging, etc.) |
| `forest_loss_ha` | Forest loss in hectares |
| `annual_rainfall_mm` | Annual rainfall |
| `avg_temp_c` | Average temperature |
| `ndvi_before`, `ndvi_after` | Vegetation index before and after the event |
| `soil_carbon_tonne_ha` | Soil carbon stock |
| `slope_degrees`, `elevation_m` | Topographic features |
| `dist_to_road_km`, `dist_to_settlement_km` | Proximity to infrastructure |
| `population_density` | Local population density |
| `protected_area` | Whether the area is a protected zone |
| `fire_incidents` | Number of fire incidents |
| `carbon_emissions_tco2e` | Carbon emissions (tCO2e) |
| `is_hotspot` | **Target** — hotspot classification label |

No missing values and no duplicate rows were found in the dataset.

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
  - `RandomForestClassifier`
  - `train_test_split`, `StandardScaler`, `LabelEncoder`
  - `RandomizedSearchCV`, `GridSearchCV`
  - Classification metrics (`accuracy`, `precision`, `recall`, `F1`, `ROC-AUC`, confusion matrix)
- **Environment:** Jupyter Notebook

---

## 🧭 Project Workflow

1. **Data Loading & Inspection** — shape, head/tail, dtypes, summary statistics, duplicate/null checks.
2. **Exploratory Data Analysis (EDA)** — target imbalance, forest loss & carbon emission distributions, regional/biome frequency, correlation heatmap, NDVI shift, spatial plots, temporal trend, topography and rainfall analysis.
3. **Preprocessing** — label encoding of categorical columns, feature/target split, feature scaling with `StandardScaler`.
4. **Train/Test Split** — 80/20 split (`random_state=42`).
5. **Baseline Model** — `RandomForestClassifier` (default parameters) trained and evaluated.
6. **Hyperparameter Tuning** — `RandomizedSearchCV` (broad search) followed by `GridSearchCV` (focused search), optimized for ROC-AUC with 3-fold cross-validation.
7. **Final Evaluation** — best tuned model evaluated on the hold-out test set.
8. **Residual & Diagnostic Analysis** — predicted-probability residuals, residual-vs-probability plot, ROC curve, and final confusion matrix.

---

## 📊 Model Performance

### Baseline Random Forest (default parameters)

| Metric | Score |
|---|---|
| Accuracy | 97.83% |
| Precision | 98.10% |
| Recall | 99.26% |
| F1 Score | 98.68% |
| ROC-AUC | 0.954 |

### Tuned Random Forest (RandomizedSearchCV → GridSearchCV)

**Best parameters:** `n_estimators=150`, `max_depth=None`, `criterion='entropy'`

| Metric | Score |
|---|---|
| Accuracy | 98.10% |
| Precision | 98.30% |
| Recall | 99.39% |
| F1 Score | 98.84% |
| ROC-AUC | 95.92% |

Hyperparameter tuning improved performance slightly across all metrics over the baseline model.

---

## 🗂️ Project Structure

```
deforestation-hotspot-classifier/
│
├── Deforestation_hotspot_RT_Classifier.ipynb   # Main notebook (EDA + modeling)
├── deforestation_hotspot_dataset.csv           # Dataset (add your own copy)
├── README.md                                   # Project documentation
└── requirements.txt                             # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/deforestation-hotspot-classifier.git
cd deforestation-hotspot-classifier
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `deforestation_hotspot_dataset.csv` in the project root (same folder as the notebook).

### 5. Run the notebook
```bash
jupyter notebook Deforestation_hotspot_RT_Classifier.ipynb
```

---

## 📦 requirements.txt

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🔮 Future Improvements

- Address the target class imbalance (e.g., SMOTE, class weighting).
- Experiment with additional models (XGBoost, LightGBM) for comparison.
- Add SHAP/feature-importance analysis for model interpretability.
- Build a lightweight API or dashboard for real-time hotspot prediction.
- Incorporate satellite-derived features for improved spatial accuracy.

---

## 🙋 Author

Maintained by the project owner. Contributions, issues, and feature requests are welcome — feel free to open an issue or submit a pull request.
