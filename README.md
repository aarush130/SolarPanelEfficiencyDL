# ☀️ Solar Panel Efficiency Prediction using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A comprehensive deep learning solution for predicting solar panel efficiency based on environmental and operational parameters.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Architecture](#-model-architecture) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Web Application](#-web-application)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project implements a **Deep Learning-based prediction system** for solar panel efficiency. By analyzing various environmental factors and panel characteristics, the model provides accurate efficiency predictions that can be used for:

- **Energy Forecasting**: Predict expected power output
- **Maintenance Planning**: Identify factors affecting efficiency
- **System Optimization**: Optimize panel configurations
- **Performance Monitoring**: Track real-time efficiency metrics

---

## ✨ Features

### 🧠 Advanced Deep Learning Models
- **Standard Neural Network**: Multi-layer feedforward architecture
- **Deep Residual Network**: Skip connections for better gradient flow
- **Attention Network**: Feature importance learning
- **Ensemble Model**: Combining multiple architectures

### 📊 Data Processing
- Comprehensive data preprocessing pipeline
- Automatic feature engineering
- Multiple scaling options (Standard, MinMax, Robust)
- Handling of missing values

### 🖥️ Beautiful Web Interface
- Modern, responsive Streamlit application
- Real-time predictions with visualization
- Interactive data exploration
- Model performance dashboard

### 📈 Visualization
- Training progress monitoring
- Feature importance analysis
- Correlation heatmaps
- Prediction vs actual plots

---

## 📁 Project Structure

```
SolarPanelEfficiencyDL/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 app.py                       # Streamlit web application
│
├── 📂 src/                         # Source code
│   ├── 📄 __init__.py             # Package initialization
│   ├── 📄 data_generator.py       # Synthetic data generation
│   ├── 📄 preprocessing.py        # Data preprocessing utilities
│   ├── 📄 model.py                # Deep learning architectures
│   └── 📄 train.py                # Training pipeline
│
├── 📂 data/                        # Dataset files
│   ├── 📄 train_data.csv          # Training dataset
│   ├── 📄 val_data.csv            # Validation dataset
│   ├── 📄 test_data.csv           # Test dataset
│   └── 📄 preprocessor.joblib     # Saved preprocessor
│
├── 📂 models/                      # Trained models
│   ├── 📄 best_model.keras        # Best checkpoint
│   ├── 📄 final_model.keras       # Final trained model
│   ├── 📄 metrics.json            # Evaluation metrics
│   └── 📄 training_results.png    # Results visualization
│
├── 📂 notebooks/                   # Jupyter notebooks
│   └── 📄 exploration.ipynb       # Data exploration notebook
│
├── 📂 logs/                        # TensorBoard logs
│
└── 📂 assets/                      # Images and resources
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/SolarPanelEfficiencyDL.git
cd SolarPanelEfficiencyDL
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Generate Dataset
```bash
python src/data_generator.py
```
This creates synthetic training, validation, and test datasets.

### 2. Train the Model
```bash
# Train with default settings (Deep Residual Network)
python src/train.py

# Train with specific architecture
python src/train.py --model-type attention --epochs 150 --batch-size 64

# Available model types: standard, deep, attention, ensemble
```

### 3. Run Web Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`

### Command Line Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--model-type` | Model architecture | `deep` |
| `--epochs` | Maximum training epochs | `100` |
| `--batch-size` | Training batch size | `32` |
| `--learning-rate` | Initial learning rate | `0.001` |
| `--patience` | Early stopping patience | `15` |

---

## 🧠 Model Architecture

### Deep Residual Network (Default)
```
Input (17 features)
    ↓
Dense (128 units, ReLU)
    ↓
[Residual Block × 4]
    │   ├── Dense (128)
    │   ├── BatchNorm
    │   ├── Dropout (0.2)
    │   ├── Dense (128)
    │   ├── BatchNorm
    │   └── Skip Connection
    ↓
Dense (64 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output (1 unit, Linear)
```

### Model Parameters
- **Total Parameters**: ~100K
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: L2 (0.001) + Dropout (0.2)

---

## 📊 Dataset

### Input Features (10 Primary)

| Feature | Description | Range |
|---------|-------------|-------|
| `solar_irradiance` | Solar radiation (W/m²) | 100 - 1200 |
| `ambient_temperature` | Air temperature (°C) | -10 to 45 |
| `panel_temperature` | Panel surface temp (°C) | -5 to 80 |
| `humidity` | Relative humidity (%) | 20 - 95 |
| `wind_speed` | Wind speed (m/s) | 0 - 15 |
| `dust_accumulation` | Dust coverage (%) | 0 - 50 |
| `panel_age` | Installation age (years) | 0 - 25 |
| `tilt_angle` | Panel tilt (degrees) | 10 - 50 |
| `cloud_cover` | Sky coverage (%) | 0 - 100 |
| `hour_of_day` | Time of day (hour) | 6 - 20 |

### Engineered Features (7 Additional)

| Feature | Description |
|---------|-------------|
| `temp_difference` | Panel temp - Ambient temp |
| `irradiance_temp_ratio` | Irradiance / (Temp + 273.15) |
| `effective_irradiance` | Irradiance adjusted for cloud/dust |
| `is_peak_hours` | Binary: 10 AM - 3 PM |
| `optimal_conditions` | Binary: Ideal conditions |
| `panel_age_category` | Categorical: 0-3 |
| `wind_cooling_factor` | Wind effect on temperature |

### Target Variable
- **`efficiency`**: Solar panel efficiency (0-25%)

---

## 📈 Results

### Model Performance Metrics

| Metric | Value |
|--------|-------|
| **MAE** | < 0.5% |
| **RMSE** | < 0.7% |
| **R² Score** | > 0.95 |
| **MAPE** | < 5% |

### Training Visualization

The training process generates comprehensive visualizations including:
- Training & validation loss curves
- Actual vs predicted scatter plots
- Error distribution histograms
- Residual analysis plots

---

## 🖥️ Web Application

The Streamlit application provides four main sections:

### 1. 🔮 Prediction Tab
- Interactive sliders for input parameters
- Real-time efficiency prediction
- Gauge visualization
- Optimization recommendations

### 2. 📊 Data Analysis Tab
- Feature distributions
- Correlation heatmaps
- Scatter plot analysis
- Raw data preview

### 3. 📈 Model Performance Tab
- Training metrics display
- Learning curves
- Model information
- Results visualization

### 4. 📚 About Tab
- Project documentation
- Feature descriptions
- Architecture details
- Quick start guide

---

## 🔧 Configuration

### Customize Training
Edit `src/train.py` to modify:
- Learning rate schedule
- Early stopping criteria
- Model hyperparameters
- Data augmentation

### Customize Data Generation
Edit `src/data_generator.py` to modify:
- Dataset size
- Feature distributions
- Physics-based efficiency formula
- Noise parameters

---

## 📝 API Reference

### DataPreprocessor
```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(scaler_type='standard')
X_scaled, y_scaled = preprocessor.fit_transform(X, y)
```

### Model Creation
```python
from src.model import create_model

model, factory = create_model(
    input_dim=17,
    model_type='deep'  # 'standard', 'attention', 'ensemble'
)
```

### Training Pipeline
```python
from src.train import train_pipeline

results = train_pipeline(
    model_type='deep',
    epochs=100,
    batch_size=32
)
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Streamlit team for the web application framework
- Scientific community for solar panel efficiency research

---

<div align="center">

**Built with ❤️ for Final Semester Project**

**Solar Panel Efficiency Research Team © 2024**

</div>
