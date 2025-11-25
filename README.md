## 📊 Overview

PatrolIQ is an intelligent crime analysis system that analyzes 500,000 crime records using machine learning to identify patterns and optimize police resource allocation.

### Key Features

- 🗺️ **Crime Hotspot Detection** - K-Means, DBSCAN, Hierarchical clustering
- ⏰ **Temporal Pattern Analysis** - Hourly, daily, seasonal trends
- 🔍 **Dimensionality Reduction** - PCA, t-SNE, UMAP visualizations
- 📈 **Model Performance** - Algorithm comparison and validation

## 🎯 Results

- **8 distinct crime zones** identified
- **75% variance** explained in 3 PCA components
- **0.58 silhouette score** achieved (K-Means)
- **500K records** processed in real-time

## 🛠️ Technologies

- Python 3.8+
- Streamlit
- Scikit-learn
- MLflow
- Plotly
- Folium

## 📁 Project Structure

```
PatrolIQ/
├── streamlit_app/
│   ├── Home.py
│   └── pages/
│       ├── 1_📊_Data_Overview.py
│       ├── 2_🗺️_Crime_Hotspots.py
│       ├── 3_⏰_Temporal_Patterns.py
│       └── 4_🔍_Model_Analysis.py
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   └── dimensionality.py
├── data/
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Pooja-p18/PatrolIQ.git
cd PatrolIQ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download Chicago Crime Dataset from [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
2. Save to `data/raw/chicago_crimes.csv`

### Run Pipeline

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/clustering.py
python src/dimensionality.py
```

### Launch App

```bash
streamlit run streamlit_app/Home.py
```

Open browser to `http://localhost:8501`

## 📊 Screenshots

### Home Page
![Home Page](screenshots/home.png)

### Crime Hotspots
![Crime Hotspots](screenshots/hotspots.png)

### Temporal Patterns
![Temporal Patterns](screenshots/temporal.png)

## 🎓 Learning Outcomes

- End-to-end ML pipeline development
- Unsupervised learning (clustering)
- Dimensionality reduction techniques
- Interactive web application development
- MLflow experiment tracking
- Cloud deployment

## 📈 Model Performance

| Algorithm | Silhouette Score | Davies-Bouldin | Clusters |
|-----------|------------------|----------------|----------|
| K-Means | 0.5823 | 0.8234 | 8 |
| DBSCAN | 0.4567 | N/A | 12 |
| Hierarchical | 0.5456 | 0.9123 | 8 |

## 💼 Business Applications

- **Police Departments**: Optimize patrol deployment
- **City Administration**: Evidence-based policy decisions
- **Analytics Firms**: Multi-city crime intelligence
- **Emergency Services**: Risk-based resource allocation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- LinkedIn: [linkedin.com/in/pooja-parashuram]
- GitHub: [Pooja Parashuram](https://github.com/Pooja-p18)
- Email: poojapbvdee@gmail.com

## 🙏 Acknowledgments

- Chicago Data Portal for the crime dataset
- Streamlit for the amazing framework
- Scikit-learn community

---

**⭐ If you find this project helpful, please give it a star!**