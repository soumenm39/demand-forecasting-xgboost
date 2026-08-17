# Demand Forecasting Using XGBoost

An end-to-end machine learning project for predicting product demand using **XGBoost Regression**. The project covers exploratory data analysis, feature selection, categorical encoding, model training, hyperparameter optimization, evaluation, model serialization, and deployment through an interactive **Streamlit** web application.

## Project Overview

Accurate demand forecasting can help businesses make better decisions related to inventory management, pricing, promotions, and supply planning.

In this project, an XGBoost regression model is trained to predict product demand from a selected set of business and market-related features.

The original dataset contains **76,000 observations** and includes information such as date, store, product, category, region, inventory, units sold, pricing, promotions, weather, seasonality, epidemic status, and demand.

The final prediction model uses six features:

* Price
* Discount
* Inventory Level
* Promotion
* Competitor Pricing
* Category

The target variable is:

* **Demand**

## Project Pipeline

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Selection
     │
     ▼
Categorical Encoding
     │
     ▼
Train/Test Split
     │
     ▼
XGBoost Regression
     │
     ▼
RandomizedSearchCV
     │
     ▼
Model Evaluation
     │
     ▼
Model Serialization
     │
     ▼
Streamlit Deployment
```

## Key Features

* Exploratory data analysis using Pandas, Matplotlib, and Seaborn
* Data quality checks for missing values and duplicates
* Feature selection for demand prediction
* Categorical encoding using `LabelEncoder`
* XGBoost regression for nonlinear demand prediction
* Hyperparameter optimization using `RandomizedSearchCV`
* 3-fold cross-validation during hyperparameter search
* Model evaluation using RMSE
* Feature-importance analysis
* Serialized model and encoder for deployment
* Interactive Streamlit prediction application

## Dataset

The dataset contains 76,000 records and 16 variables.

Important variables include:

| Variable           | Description             |
| ------------------ | ----------------------- |
| Date               | Observation date        |
| Store ID           | Store identifier        |
| Product ID         | Product identifier      |
| Category           | Product category        |
| Region             | Geographic region       |
| Inventory Level    | Available inventory     |
| Units Sold         | Number of units sold    |
| Units Ordered      | Number of units ordered |
| Price              | Product price           |
| Discount           | Discount percentage     |
| Weather Condition  | Weather condition       |
| Promotion          | Promotion indicator     |
| Competitor Pricing | Competitor price        |
| Seasonality        | Seasonal condition      |
| Epidemic           | Epidemic indicator      |
| Demand             | Target variable         |

The dataset used for this project was obtained from the publicly available Demand Forecasting Dataset repository.

## Exploratory Data Analysis

The EDA workflow includes:

* Dataset structure and data types
* Missing-value analysis
* Duplicate-value analysis
* Descriptive statistics
* Distribution analysis
* Categorical-variable exploration
* Relationship between input variables and demand
* Feature-importance visualization

The dataset contains no missing values and no duplicate rows in the analyzed dataset.

## Machine Learning Model

### Algorithm

The project uses:

```text
XGBRegressor
```

with the objective:

```text
reg:squarederror
```

The dataset is divided into:

```text
Training set: 80%
Testing set: 20%
```

A fixed random state of `42` is used for reproducibility.

### Hyperparameter Optimization

`RandomizedSearchCV` is used to search over:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
gamma
min_child_weight
```

The search evaluates 25 randomly selected parameter combinations using 3-fold cross-validation.

### Best Parameters

The best-performing parameter combination found in the training notebook was:

```text
n_estimators      = 200
max_depth         = 8
learning_rate     = 0.05
subsample         = 0.8
colsample_bytree  = 1.0
gamma             = 0.3
min_child_weight  = 5
```

## Model Performance

The final XGBoost model achieved:

| Metric    |          Result |
| --------- | --------------: |
| Test RMSE | **35.50 units** |

The test set contains 15,200 observations.

> RMSE represents the typical magnitude of prediction error in the same units as the target variable, although it is sensitive to larger errors.

## Feature Importance

The trained model reports the following feature importances:

| Feature            | Importance |
| ------------------ | ---------: |
| Promotion          |     0.5845 |
| Category           |     0.2770 |
| Price              |     0.0730 |
| Competitor Pricing |     0.0243 |
| Discount           |     0.0210 |
| Inventory Level    |     0.0202 |

In this trained model, **Promotion** has the largest feature importance, followed by **Category**.

These values represent the model's internal feature importance and should not be interpreted directly as causal effects.

## Streamlit Application

The project includes an interactive Streamlit application for making demand predictions.

The application allows users to enter:

* Price
* Discount
* Inventory Level
* Promotion
* Competitor Pricing
* Category

The trained XGBoost model and categorical encoder are loaded from the `models/` directory.

The application then generates a predicted demand value in units.

### Run the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Repository Structure

```text
demand-forecasting-xgboost/
│
├── app/
│   └── app.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
│
├── models/
│   ├── Xgboost_demand_model.pkl
│   └── label_encoder.pkl
│
├── data/
│   ├── demand_forecasting.csv
│   └── data.txt
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Deployment

* Streamlit

### Model Persistence

* Pickle

## Installation

Clone the repository:

```bash
git clone https://github.com/soumenm39/demand-forecasting-xgboost.git
```

Move into the project directory:

```bash
cd demand-forecasting-xgboost
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/01_eda.ipynb
```

for exploratory data analysis and:

```text
notebooks/02_model_training.ipynb
```

for model development and evaluation.

## Reproducibility

The project uses fixed random states during data splitting and hyperparameter optimization:

```python
random_state=42
```

The trained model and label encoder are included in the `models/` directory so that the Streamlit application can perform inference without retraining the model.

## Future Improvements

Potential improvements to this project include:

* Building a complete Scikit-learn preprocessing and modeling pipeline
* Comparing XGBoost with Random Forest, LightGBM, CatBoost, and linear regression
* Adding MAE and R² to the model evaluation dashboard
* Performing time-aware train/test splitting
* Adding temporal features such as month, week, day-of-week, and lagged demand
* Adding model explainability using SHAP
* Containerizing the application with Docker
* Adding automated model training and testing with CI/CD
* Deploying the Streamlit application to a cloud platform
* Adding experiment tracking using MLflow

## Author

**Soumen Mondal**

Research Scholar | Machine Learning & Computational Science

GitHub: [@soumenm39](https://github.com/soumenm39)

## License

This project is intended for educational and portfolio purposes. Please refer to the dataset's original source and licensing terms before redistributing the dataset.
