# ☀️ Solar Panel Efficiency Prediction using Deep Learning

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aarush130/SolarPanelEfficiency)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

**🚀 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/aarush130/SolarPanelEfficiency)**

A deep learning-based web application for predicting solar panel efficiency across India, featuring real data for 50+ cities.

---

## 🎯 Features

- **🇮🇳 India Solar Map** - Interactive visualization of solar potential across 50+ Indian cities
- **🔮 Efficiency Prediction** - Real-time predictions based on environmental parameters
- **📊 Data Analysis** - Comprehensive charts, correlations, and state-wise comparisons
- **🏆 City Rankings** - Find the best locations for solar installations in any state

---

## 📸 Screenshots

### India Solar Map
![India Solar Map](screenshots/india_map.png)

### Efficiency Prediction
![Prediction Interface](screenshots/prediction.png)

### Data Analysis
![Data Analysis](screenshots/analysis.png)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: Pandas, NumPy
- **Deep Learning**: TensorFlow/Keras (optional)

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/aarush130/SolarPanelEfficiencyDL.git
cd SolarPanelEfficiencyDL

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### View Online

**[🤗 Open on Hugging Face Spaces](https://huggingface.co/spaces/aarush130/SolarPanelEfficiency)**

---

## 📊 Data Sources

- Ministry of New and Renewable Energy (MNRE)
- India Meteorological Department (IMD)
- National Institute of Solar Energy (NISE)

---

## 🏆 Top Solar Locations in India

| Rank | City | State | GHI (kWh/m²/day) |
|------|------|-------|------------------|
| 1 | Leh | Ladakh | 5.90 |
| 2 | Jaisalmer | Rajasthan | 5.89 |
| 3 | Jodhpur | Rajasthan | 5.85 |
| 4 | Kutch | Gujarat | 5.82 |
| 5 | Bikaner | Rajasthan | 5.80 |

---

## 📁 Project Structure

```
SolarPanelEfficiencyDL/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_generator.py  # Synthetic data generation
│   ├── preprocessing.py   # Data preprocessing
│   ├── model.py          # Deep learning models
│   ├── train.py          # Training pipeline
│   └── utils.py          # Utility functions
├── notebooks/
│   └── exploration.ipynb  # Data exploration notebook
└── README.md
```

---

## 👨‍💻 Author

**Aarush Saxena**  
VIT University  
Final Semester Project - B.Tech

---

## 📄 License

MIT License - Feel free to use this project for learning and research.

---

<p align="center">
  Built with ❤️ using Streamlit & Plotly
</p>
