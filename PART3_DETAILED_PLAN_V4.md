# Part 3 - Experiments

## Part 3.1 - Feature engineering

### Cell: Markdown - Part 3.1 Introduction
```markdown
### Part 3.1 - Feature engineering

Feature engineering is the process of transforming raw data into features that better represent the underlying problem. For our diabetes dataset, we'll systematically test different preprocessing approaches to see which ones improve model performance.

**The feature engineering we'll test:**

1. **Scaling Methods**: Different algorithms respond differently to feature scaling:
   - **StandardScaler**: Centers data around 0 with standard deviation of 1
   - **No scaling**: Use data as-is (baseline)

2. **Feature Selection**: Using Pearson correlation to remove highly correlated features:
   - **With feature selection**: Remove highly correlated features (threshold > 0.8, like s1 and s2)
   - **Without feature selection**: Use all features (baseline)

This gives us 2×2 = 4 different feature engineering configurations. Each will be tested with both Linear Regression and KNN to find the optimal preprocessing approach.
```

### Cell: Python - Feature Engineering Pipeline
```python
# Feature Engineering Pipeline

def apply_feature_engineering(config, X_train, X_val, y_train):
    """
    Apply feature engineering based on configuration
    Returns processed training and validation sets
    """
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    
    # 1. Feature selection using Pearson correlation (remove highly correlated features)
    if config['feature_selection'] == 'remove_correlated':
        X_train_processed, removed_features = remove_highly_correlated_features(X_train_processed, y_train)
        X_val_processed = X_val_processed.drop(columns=removed_features)
        print(f"Removed highly correlated features: {removed_features}")
    
    # 2. Scaling
    if config['scaling'] == 'standard':
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train_processed)
        X_val_processed = scaler.transform(X_val_processed)
    # No scaling for 'none'
    
    return X_train_processed, X_val_processed

def remove_highly_correlated_features(X, y, threshold=0.8):
    """
    Remove features with high correlation, keeping the one more correlated with target
    """
    corr_matrix = X.corr().abs()
    target_corr = X.corrwith(y).abs()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
    
    features_to_remove = set()
    for feat1, feat2, corr in high_corr_pairs:
        if target_corr[feat1] > target_corr[feat2]:
            features_to_remove.add(feat2)
        else:
            features_to_remove.add(feat1)
    
    return X.drop(columns=list(features_to_remove)), list(features_to_remove)

print("Feature engineering pipeline ready")
```

### Cell: Python - Feature Engineering Configurations
```python
# Define feature engineering configurations (simplified for group of 2)
feature_configs = [
    # Baseline
    {'scaling': 'none', 'feature_selection': 'none'},
    
    # Scaling only
    {'scaling': 'standard', 'feature_selection': 'none'},
    
    # Feature selection only
    {'scaling': 'none', 'feature_selection': 'remove_correlated'},
    
    # Both scaling and feature selection
    {'scaling': 'standard', 'feature_selection': 'remove_correlated'},
]

print(f"Total feature engineering configurations: {len(feature_configs)}")
for i, config in enumerate(feature_configs):
    print(f"Config {i+1}: {config}")
```

## Part 3.2 - Model and Hyperparameter Experiments

### Cell: Markdown - Part 3.2 Introduction
```markdown
### Part 3.2 - Model and Hyperparameter Experiments

Now we'll test different machine learning algorithms with various hyperparameter settings. We'll compare Linear Regression against K-Nearest Neighbors (KNN) to demonstrate why Linear Regression is better suited for this diabetes prediction problem.

**Linear Regression Configurations:**
- **fit_intercept**: [True, False] - Tests whether we need to calculate an intercept term or if the data is already centered

**K-Nearest Neighbors Configurations:**
- **n_neighbors**: [3, 15] - Tests different neighborhood sizes (small vs large)
- **weights**: ['uniform', 'distance'] - Tests different ways to weight the neighbors

This gives us 2 Linear Regression configurations and 4 KNN configurations (2×2), for a total of 6 model configurations. Combined with our 4 feature engineering approaches, we'll run 24 total experiments.

**Why we expect Linear Regression to win:**
1. **Linear relationships**: Our EDA showed clear linear patterns between features and diabetes progression
2. **Small dataset**: With only 353 samples, simpler models like Linear Regression are more stable
3. **Preprocessed features**: The diabetes data is already scaled and normalized
4. **Medical interpretability**: Linear Regression coefficients tell us exactly how much each factor contributes to diabetes risk
```

### Cell: Python - Model Configurations
```python
# Define model configurations
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

model_configs = [
    # Linear Regression configurations
    {'model': 'linear_regression', 'fit_intercept': True},
    {'model': 'linear_regression', 'fit_intercept': False},
    
    # KNN configurations
    {'model': 'knn', 'n_neighbors': 3, 'weights': 'uniform'},
    {'model': 'knn', 'n_neighbors': 3, 'weights': 'distance'},
    {'model': 'knn', 'n_neighbors': 15, 'weights': 'uniform'},
    {'model': 'knn', 'n_neighbors': 15, 'weights': 'distance'},
]

def create_model(config):
    """
    Create model instance based on configuration
    """
    if config['model'] == 'linear_regression':
        return LinearRegression(fit_intercept=config['fit_intercept'])
    elif config['model'] == 'knn':
        return KNeighborsRegressor(
            n_neighbors=config['n_neighbors'],
            weights=config['weights']
        )

print(f"Total model configurations: {len(model_configs)}")
for i, config in enumerate(model_configs):
    print(f"Model {i+1}: {config}")
```

## Part 3.3 - Quality Metric Usage

### Cell: Markdown - Part 3.3 Introduction
```markdown
### Part 3.3 - Quality Metric Usage

For this regression problem, we need to choose the right quality metric to evaluate our models. Since we're predicting diabetes progression (a continuous value), we'll use R² (R-squared) as our primary metric.

**Why R² is perfect for this problem:**
- R² measures how much of the variance in diabetes progression is explained by our model
- It ranges from 0 to 1, making it easy to interpret
- Values close to 1 mean our model explains most of the variation in the data
- Values close to 0 mean our model isn't much better than just predicting the average

**How we'll use R²:**
1. **During experiments**: 5-fold cross-validation to compare different model configurations
2. **Final evaluation**: Test set evaluation to estimate real-world performance

This will help us objectively determine whether Linear Regression or KNN performs better for predicting diabetes progression, and which feature engineering approaches work best.
```

### Cell: Python - R² Setup
```python
# Setup quality metric for experiments
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y, kfold):
    """
    Evaluate model using R² with cross-validation
    Returns mean CV score and standard deviation
    Note: Using 5-fold CV due to small sample size (352 samples)
    """
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    return cv_scores.mean(), cv_scores.std()

def final_evaluation(model, X_test, y_test):
    """
    Final evaluation on test set with comprehensive metrics
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred
    }

print("Quality metric setup complete - using R² for model evaluation")
```

## Part 3.4 - Grid-Search K-Fold Cross-Validation with Experiments

### Cell: Markdown - Part 3.4 Introduction
```markdown
### Part 3.4 - Grid-Search K-Fold Cross-Validation with Experiments

Now we'll systematically test all combinations using 5-fold cross-validation. This is the heart of our experimental process where we'll run all 24 combinations (4 feature configs × 6 model configs) to find the best performing setup.

**Our systematic approach:**
1. **Split training data**: We'll split our training data into train/validation sets for each experiment
2. **Apply feature engineering**: Each configuration will be applied to both training and validation sets
3. **Cross-validation**: We'll use 5-fold CV on the training portion to get robust performance estimates
4. **Validation**: We'll also test on the held-out validation set to prevent overfitting
5. **Record results**: All 24 combinations will be evaluated and compared

**Why this approach is robust:**
- **No data leakage**: Test set remains completely untouched until final evaluation
- **Fair comparison**: All models are evaluated using the same methodology
- **Statistical significance**: 5-fold CV gives us confidence in our results
- **Overfitting prevention**: Validation set helps us select the best generalizing model

We'll use the validation scores to select the best configuration, then train the final model on the full training set and evaluate on the test set.
```

### Cell: Python - Data Preparation Setup
```python
# Data preparation for experiments
# Note: We only have access to training set for now, test set will be used only in Part 5
# Split the training data into features and target
X_train = train_data_cleaned.drop('target', axis=1)
y_train = train_data_cleaned['target']

print(f"Training data shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"Note: Test set will be used only in Part 5 for final evaluation")
```

### Cell: Python - Experiment Execution
```python
# Execute all experiments using 5-fold cross-validation
from sklearn.model_selection import KFold
import pandas as pd

# Use 5-fold cross-validation on training set (test set remains untouched until Part 5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"Using 5-fold cross-validation on training set:")
print(f"Training set shape: {X_train.shape}")
print(f"Each fold: 80% train (~282 samples), 20% validation (~71 samples)")
print(f"Note: Test set will be used only in Part 5 for final evaluation")

# Run all experiments
results = []
experiment_count = 0
total_experiments = len(feature_configs) * len(model_configs)

for feature_config in feature_configs:
    for model_config in model_configs:
        experiment_count += 1
        print(f"\nExperiment {experiment_count}/{total_experiments}")
        print(f"Feature config: {feature_config}")
        print(f"Model config: {model_config}")
        
        try:
            # Apply feature engineering to full training set
            X_processed, _ = apply_feature_engineering(
                feature_config, X_train, X_train, y_train
            )
            
            # Create model
            model = create_model(model_config)
            
            # 5-fold cross-validation
            cv_mean, cv_std = evaluate_model(model, X_processed, y_train, kfold)
            
            # Store results
            result = {
                'experiment_id': experiment_count,
                'feature_config': feature_config,
                'model_config': model_config,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'feature_count': X_processed.shape[1]
            }
            results.append(result)
            
            print(f"CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
            
        except Exception as e:
            print(f"Error in experiment {experiment_count}: {e}")
            continue

print(f"\nCompleted {len(results)} experiments successfully")
```

### Cell: Python - Results Analysis
```python
# Analyze experiment results
results_df = pd.DataFrame(results)

# Sort by CV score (best first)
results_df_sorted = results_df.sort_values('cv_mean', ascending=False)

print("Top 10 Experiment Results:")
print("=" * 80)
for i, row in results_df_sorted.head(10).iterrows():
    print(f"Rank {results_df_sorted.index.get_loc(i)+1}:")
    print(f"  Feature Config: {row['feature_config']}")
    print(f"  Model Config: {row['model_config']}")
    print(f"  CV R²: {row['cv_mean']:.4f} ± {row['cv_std']:.4f}")
    print(f"  Features: {row['feature_count']}")
    print()

# Find best configuration
best_result = results_df_sorted.iloc[0]
print(f"BEST CONFIGURATION:")
print(f"Feature Engineering: {best_result['feature_config']}")
print(f"Model: {best_result['model_config']}")
print(f"CV R²: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
```

### Cell: Python - Results Visualization
```python
# Create comprehensive results visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Validation scores comparison
val_scores = results_df_sorted['val_score']
experiment_labels = [f"Exp {i+1}" for i in range(len(val_scores))]

axes[0,0].bar(range(len(val_scores)), val_scores, color='skyblue', alpha=0.7)
axes[0,0].set_title('Validation R² Scores (All Experiments)')
axes[0,0].set_xlabel('Experiment Rank')
axes[0,0].set_ylabel('R² Score')
axes[0,0].axhline(y=val_scores.mean(), color='red', linestyle='--', label=f'Mean: {val_scores.mean():.3f}')
axes[0,0].legend()

# 2. Model type comparison
model_types = results_df['model_config'].apply(lambda x: x['model'])
model_scores = results_df.groupby(model_types)['val_score'].mean()

axes[0,1].bar(model_scores.index, model_scores.values, color=['green', 'orange'])
axes[0,1].set_title('Average R² by Model Type')
axes[0,1].set_ylabel('Average R² Score')

# 3. Feature engineering impact
scaling_impact = results_df.groupby(results_df['feature_config'].apply(lambda x: x['scaling']))['val_score'].mean()
axes[1,0].bar(scaling_impact.index, scaling_impact.values, color='lightcoral')
axes[1,0].set_title('Average R² by Scaling Method')
axes[1,0].set_ylabel('Average R² Score')

# 4. CV vs Validation scores
axes[1,1].scatter(results_df['cv_mean'], results_df['val_score'], alpha=0.6)
axes[1,1].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
axes[1,1].set_xlabel('CV R² Score')
axes[1,1].set_ylabel('Validation R² Score')
axes[1,1].set_title('CV vs Validation R² Scores')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Summary statistics
print("EXPERIMENT SUMMARY:")
print(f"Total experiments: {len(results_df)}")
print(f"Best validation R²: {results_df['val_score'].max():.4f}")
print(f"Worst validation R²: {results_df['val_score'].min():.4f}")
print(f"Average validation R²: {results_df['val_score'].mean():.4f}")
print(f"Standard deviation: {results_df['val_score'].std():.4f}")
```

## Part 3.5 - Final Model Training and Test Evaluation

### Cell: Markdown - Part 3.5 Introduction
```markdown
### Part 3.5 - Final Model Training and Test Evaluation

Now we'll train the best model configuration on the full training set and evaluate it on the test set. This is the only time we'll touch the test data, ensuring no data leakage.

The best configuration from our experiments will be used to:
1. Apply the optimal feature engineering to full training set
2. Train the final model on all training data
3. Evaluate on the test set for final performance estimation
4. Generate comprehensive performance metrics and visualizations

This final evaluation will demonstrate why our chosen model and feature engineering approach is superior for predicting diabetes progression.
```

### Cell: Python - Final Model Training
```python
# Train final model on full training set
print("Training final model on full training set...")
print(f"Best configuration: {best_result['feature_config']} + {best_result['model_config']}")

# Apply best feature engineering to full training set
X_train_final, X_test_final = apply_feature_engineering(
    best_result['feature_config'], X_train, X_test, y_train
)

# Train final model
final_model = create_model(best_result['model_config'])
final_model.fit(X_train_final, y_train)

# Evaluate on test set
test_results = final_evaluation(final_model, X_test_final, y_test)

print(f"\nFINAL TEST SET PERFORMANCE:")
print(f"R² Score: {test_results['r2']:.4f}")
print(f"RMSE: {test_results['rmse']:.4f}")
print(f"MAE: {test_results['mae']:.4f}")
print(f"MSE: {test_results['mse']:.4f}")
```

### Cell: Python - Final Results Visualization
```python
# Create final results visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predictions vs Actual
axes[0,0].scatter(y_test, test_results['predictions'], alpha=0.6, color='blue')
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect prediction')
axes[0,0].set_xlabel('Actual Values')
axes[0,0].set_ylabel('Predicted Values')
axes[0,0].set_title(f'Predictions vs Actual (R² = {test_results["r2"]:.4f})')
axes[0,0].legend()

# 2. Residuals plot
residuals = y_test - test_results['predictions']
axes[0,1].scatter(test_results['predictions'], residuals, alpha=0.6, color='green')
axes[0,1].axhline(y=0, color='red', linestyle='--')
axes[0,1].set_xlabel('Predicted Values')
axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title('Residuals Plot')

# 3. Feature importance (if linear regression)
if best_result['model_config']['model'] == 'linear_regression':
    feature_names = [f'Feature {i+1}' for i in range(X_train_final.shape[1])]
    coefficients = final_model.coef_
    axes[1,0].bar(feature_names, coefficients, color='orange')
    axes[1,0].set_title('Feature Coefficients (Linear Regression)')
    axes[1,0].set_ylabel('Coefficient Value')
    axes[1,0].tick_params(axis='x', rotation=45)
else:
    axes[1,0].text(0.5, 0.5, 'Feature importance\nnot available for KNN', 
                   ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Feature Importance')

# 4. Performance comparison
model_names = ['Linear Regression', 'KNN']
best_lr_score = results_df[results_df['model_config'].apply(lambda x: x['model'] == 'linear_regression')]['val_score'].max()
best_knn_score = results_df[results_df['model_config'].apply(lambda x: x['model'] == 'knn')]['val_score'].max()

axes[1,1].bar(model_names, [best_lr_score, best_knn_score], color=['green', 'orange'])
axes[1,1].set_title('Best Performance by Model Type')
axes[1,1].set_ylabel('Validation R² Score')
axes[1,1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Final summary
print(f"\nFINAL EXPERIMENT SUMMARY:")
print(f"Best model: {best_result['model_config']['model']}")
print(f"Best feature engineering: {best_result['feature_config']}")
print(f"Test set R²: {test_results['r2']:.4f}")
print(f"Model successfully demonstrates why {best_result['model_config']['model']} is superior for this dataset")
```
