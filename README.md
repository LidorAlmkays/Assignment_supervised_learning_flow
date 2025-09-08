# Supervised Learning Flow - Diabetes Progression Prediction

## 📋 Project Overview

This project implements a complete supervised learning pipeline for predicting diabetes progression using baseline medical measurements. The work demonstrates a systematic approach to machine learning from data exploration to final model evaluation, following best practices in experimental design and validation.

**Authors:** Lidor E. (2612), Amit L. (6819)  
**Course:** Machine Learning - Supervised Learning Flow Assignment  
**Dataset:** Diabetes Progression Dataset (442 patients)

## 🎯 Problem Statement

**Objective:** Predict the quantitative measure of diabetes progression one year after baseline using 10 baseline medical measurements.

**Clinical Relevance:** Early prediction of diabetes progression can help identify higher-risk patients and guide treatment decisions, making this a valuable tool for healthcare professionals.

## 📊 Dataset Description

### Dataset Characteristics
- **Total Patients:** 442 diabetes patients
- **Features:** 10 baseline medical measurements
- **Target:** Continuous measure of disease progression after one year
- **Training Set:** 353 samples
- **Test Set:** 89 samples

### Feature Information
| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Demographic |
| `sex` | Gender | Demographic |
| `bmi` | Body mass index | Physical |
| `bp` | Average blood pressure | Physical |
| `s1` | Total serum cholesterol | Blood serum |
| `s2` | Low-density lipoproteins (LDL) | Blood serum |
| `s3` | High-density lipoproteins (HDL) | Blood serum |
| `s4` | Total cholesterol / HDL ratio | Blood serum |
| `s5` | Log of serum triglycerides level | Blood serum |
| `s6` | Blood sugar level | Blood serum |

**Note:** All features are mean-centered and scaled by the standard deviation times √n_samples.

## 🏗️ Project Structure

```
Assignment_supervised_learning_flow/
├── README.md                                    # This file
├── Assignment_supervised_learning_flow.ipynb    # Main notebook
├── assignment_ml_flow_instructions_short.pdf    # Assignment instructions
└── data/
    ├── diabetes_train.csv                       # Training data (353 samples)
    ├── diabetes_test.csv                        # Test data (89 samples)
    └── diabetes_description.txt                 # Dataset documentation
```

## 🔬 Methodology

### 1. Data Loading and Exploratory Data Analysis (EDA)
- **Data Quality Assessment:** Missing values, data types, basic statistics
- **Distribution Analysis:** Histograms and boxplots for all variables
- **Outlier Treatment:** IQR-based clipping (1.5×IQR boundaries)
- **Correlation Analysis:** Feature correlation heatmap and multicollinearity detection
- **Feature-Target Relationships:** Pairwise plots to understand linear patterns

### 2. Experimental Design
**Feature Engineering Configurations:**
- No preprocessing vs. StandardScaler
- All features vs. correlation-based feature removal (threshold: 0.8)

**Model Configurations:**
- **Linear Regression:** With/without intercept
- **K-Nearest Neighbors:** k=3-15, uniform/distance weights
- **Polynomial Regression:** Degrees 1-3, with/without intercept
- **Random Forest:** 100-300 estimators, max_depth 6-10, min_samples_leaf 1-2

### 3. Validation Strategy
- **5-Fold Cross-Validation:** Systematic evaluation of all combinations
- **Primary Metric:** R² (coefficient of determination)
- **Secondary Metrics:** MSE, SSE for comprehensive evaluation
- **Grid Search:** Cartesian product of all feature × model configurations

### 4. Model Selection and Training
- **Best Configuration Selection:** Based on highest cross-validated R²
- **Final Training:** Retrain best configuration on full training set
- **Test Evaluation:** Unbiased assessment on held-out test set

## 📈 Results

### Best Model Configuration
- **Algorithm:** Linear Regression with intercept
- **Preprocessing:** No additional scaling (data already standardized)
- **Feature Selection:** All 10 features retained

### Performance Metrics
| Metric | Cross-Validation | Test Set |
|--------|------------------|----------|
| **R²** | 0.499 ± 0.088 | 0.465 |
| **MSE** | 3018 ± 765 | 2828 |
| **SSE** | 213,123 ± 54,461 | 251,700 |

### Key Findings
1. **Linear relationships dominate:** Simple linear regression outperformed complex models
2. **Feature correlation:** s1-s2 correlation (0.889) didn't significantly impact performance
3. **Generalization:** Test performance (R²=0.465) closely matches CV performance (R²=0.499)
4. **Clinical significance:** Model explains ~50% of variance in diabetes progression

## 🛠️ Technical Implementation

### Dependencies
```python
# Core libraries
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0

# Machine learning
scikit-learn >= 1.0.0

# Data handling
pathlib (built-in)
```

### Key Functions
- `apply_feature_engineering()`: Applies preprocessing configurations
- `remove_highly_correlated_features()`: Handles multicollinearity
- `create_model()`: Model factory for different algorithms
- `evaluate_model()`: Comprehensive model evaluation with CV

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see Dependencies above)

### Installation
1. Clone or download the repository
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `Assignment_supervised_learning_flow.ipynb` in Jupyter Notebook
4. Run all cells to reproduce the analysis

### Usage
The notebook is designed to run sequentially from top to bottom. Each section builds upon the previous one:

1. **Part 1:** Student details and AI assistance documentation
2. **Part 2:** Data loading and exploratory analysis
3. **Part 3:** Systematic experiments with grid search
4. **Part 4:** Final model training
5. **Part 5:** Test set evaluation and visualization

## 📊 Visualizations

The project includes comprehensive visualizations:

1. **Distribution Analysis:** Histograms and boxplots for all features
2. **Correlation Heatmap:** Feature correlation matrix
3. **Pairwise Relationships:** Feature-target scatter plots
4. **Prediction Visualization:** Actual vs. predicted values scatter plot
5. **Outlier Analysis:** Before/after outlier treatment comparison

## 🔍 Key Insights

### Data Quality
- **No missing values** in the dataset
- **Outliers present** but handled with IQR-based clipping
- **High correlation** between s1 (total cholesterol) and s2 (LDL) - medically expected

### Model Performance
- **Linear models optimal** for this dataset size and feature relationships
- **Cross-validation stable** with reasonable standard deviation
- **Good generalization** from training to test set

### Clinical Interpretation
- **50% variance explained** is clinically meaningful for medical prediction
- **Baseline measurements** are reasonably predictive of one-year progression
- **Simple model** preferred for interpretability and reliability

## 🤖 AI Assistance

This project used AI tools responsibly for learning and clarification:

- **ChatGPT:** For understanding medical terminology and ML concepts
- **Purpose:** Learning about algorithms, hyperparameters, and evaluation metrics
- **Approach:** Asked conceptual questions rather than requesting solutions
- **Documentation:** All AI interactions documented in the notebook

## 📚 References

1. **Dataset Source:** [NCSU Diabetes Dataset](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)
2. **Original Paper:** Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics
3. **Scikit-learn Documentation:** [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

## 📝 Assignment Compliance

This project fulfills all requirements of the supervised learning flow assignment:

- ✅ **Part 1 (10 pts):** Student details, AI assistance, dataset explanation
- ✅ **Part 2 (10 pts):** Data loading, comprehensive EDA with visualizations
- ✅ **Part 3 (70 pts):** Systematic experiments with grid search and cross-validation
- ✅ **Part 4 (10 pts):** Final model training on best configuration
- ✅ **Part 5 (10 pts):** Test evaluation with performance metrics and visualization

## 👥 Authors

- **Lidor E. (2612)** - Data analysis, model implementation, visualization
- **Amit L. (6819)** - Experimental design, validation, documentation

## 📄 License

This project is part of an academic assignment and is intended for educational purposes.

---

*For questions or clarifications, please refer to the detailed explanations in the Jupyter notebook or contact the authors.*
