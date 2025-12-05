# ML_00016599

# Serie A Match Result Prediction (MLDA Coursework)

This project is part of the 6COSC017C-n Machine Learning and Data Analytics coursework.
It builds a full end-to-end pipeline to predict Serie A match results (home win, draw,
away win) using historical match statistics.

## Project Structure

- `notebook.ipynb` — full data analysis, preprocessing, model training and evaluation
- `app/`
  - `Home.py` — main Streamlit app (model selection + prediction + preprocessing & evaluation info)
  - `pages/1_EDA.py` — EDA page for visualising the dataset
  - `models/models_dict.pkl` — trained models (LogReg, Random Forest, KNN)
  - `models/results.csv` — test accuracy for each model
- `data/` — raw CSV files per season (`season-*.csv`) and merged file `serie_a_merged.csv`
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT license

## Setup Instructions

1. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate (source venv/Scripts/activate , . venv/Scripts/activate)  # on Windows
   # source venv/bin/activate  # on macOS/Linux
   ```

   2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   3. run jupyter file notebook.ipynb
   4. streamlit run app/Home.py

## Technologies and Libraries

The project was developed using Python and a set of widely used data science libraries:

- **pandas** for data manipulation and merging seasonal match files
- **numpy** for numerical operations
- **matplotlib** and **seaborn** for exploratory data analysis and visualisation
- **scikit-learn** for preprocessing (scaling), feature engineering, model training, hyperparameter tuning, and evaluation
- **joblib** for serialising and loading trained machine learning models into the Streamlit application
- **streamlit** for building a multi-page interactive web app showcasing EDA, preprocessing steps, model usage, and prediction results
