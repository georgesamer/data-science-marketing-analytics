# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(filepath):
    """Load and validate the dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
def explore_data(df):
    """Perform initial data exploration"""
    print("="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Data Types & Missing Values ---")
    print(df.info())
    
    print("\n--- Statistical Summary ---")
    print(df.describe().round(2))
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n")


def visualize_relationships(df):
    """Create visualizations for feature relationships"""
    print("Generating visualizations...\n")
    
    # Pairplot for all numerical features
    print("1. Creating pairplot...")
    sns.pairplot(df, height=2.5)
    plt.suptitle('Feature Relationships', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    print("2. Creating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, fmt='.2f', square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Joint plot for key relationship
    print("3. Creating jointplot...")
    sns.jointplot(x='Time on Website', y='Yearly Amount Spent', 
                  data=df, kind='reg', height=8)
    plt.tight_layout()
    plt.show()


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data(df, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    # Define features and target
    feature_cols = ['Avg. Session Length', 'Time on App', 
                    'Time on Website', 'Length of Membership']
    
    X = df[feature_cols]
    y = df['Yearly Amount Spent']
    
    print("="*80)
    print("DATA PREPARATION")
    print("="*80)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    print(f"Split ratio: {int((1-test_size)*100)}% train, {int(test_size*100)}% test\n")
    
    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_model(X_train, y_train):
    """Train the Linear Regression model"""
    print("="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("\n✓ Model trained successfully")
    print(f"Intercept: ${model.intercept_:.2f}")
    
    return model


def visualize_feature_impact(df):
    """Visualize the impact of key features"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    features = ['Avg. Session Length', 'Time on App', 
                'Time on Website', 'Length of Membership']
    
    for idx, feature in enumerate(features):
        row, col = idx // 2, idx % 2
        sns.regplot(data=df, x=feature, y='Yearly Amount Spent', 
                    ax=axes[row, col], scatter_kws={'alpha':0.5})
        axes[row, col].set_title(f'{feature} vs Yearly Amount Spent')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MODEL EVALUATION
# ============================================================================
def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance with comprehensive metrics"""
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n--- Performance Metrics ---")
    print(f"R² Score: {r2:.4f} ({r2*100:.2f}% of variance explained)")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Mean Squared Error (MSE): ${mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    
    # Feature importance (coefficients)
    print("\n--- Feature Importance (Coefficients) ---")
    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    
    for _, row in coeff_df.iterrows():
        print(f"{row['Feature']:25s}: ${row['Coefficient']:8.2f}")
    
    print("\nInterpretation:")
    print("- A $1 increase in coefficient means that much increase in yearly spending")
    print(f"- Most impactful feature: {coeff_df.iloc[0]['Feature']}")
    
    # Residual plot
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual Yearly Amount Spent')
    axes[0].set_ylabel('Predicted Yearly Amount Spent')
    axes[0].set_title('Predicted vs Actual Values')
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return y_pred, coeff_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    # File path
    filepath = r"D:\VS_code\VS_code_WorkSpace\python_projects\real\ecommerce-customers.csv"
    
    # Load data
    df = load_data(filepath)
    if df is None:
        return
    
    # Explore data
    explore_data(df)
    
    # Visualize relationships
    visualize_relationships(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Visualize feature impacts
    visualize_feature_impact(df)
    
    # Evaluate model
    predictions, coeff_df = evaluate_model(model, X_test, y_test, feature_names)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()