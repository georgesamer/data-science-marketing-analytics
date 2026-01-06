# data-science-marketing-analytics
Complete Data Science projects for Marketing Analytics: E-commerce Customer Prediction using Linear Regression &amp; RFM Analysis with Customer Segmentation. Includes comprehensive EDA, visualization, and actionable business insights.

# 📊 Data Science for Marketing Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

A comprehensive data science project collection focused on marketing analytics, featuring customer behavior prediction and RFM (Recency, Frequency, Monetary) analysis with advanced customer segmentation.

## 🎯 Project Overview

This repository contains two major marketing analytics projects completed as part of the Udemy course "Data Science for Marketing Analytics" by Eng. Mustafa Othman.

### Projects Included:

1. **E-commerce Customer Spending Prediction**
   - Linear Regression model to predict yearly customer spending
   - Feature importance analysis
   - Comprehensive performance metrics

2. **RFM Analysis & Customer Segmentation**
   - Customer segmentation using RFM methodology
   - High-Value Customer (HVC) identification
   - Actionable business insights and recommendations

## 🚀 Features

### E-commerce Analysis
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature relationship visualization
- ✅ Linear Regression modeling
- ✅ Model performance evaluation (R², MAE, RMSE)
- ✅ Feature importance ranking
- ✅ Residual analysis

### RFM Customer Analysis
- ✅ Customer segmentation (10 segments)
- ✅ High-Value Customer identification
- ✅ Revenue contribution analysis
- ✅ Customer behavior insights
- ✅ Actionable marketing recommendations
- ✅ Interactive visualizations

## 📊 Key Metrics & Results

### Linear Regression Model Performance
- **R² Score**: ~98% variance explained
- **Features analyzed**: Session Length, App Time, Website Time, Membership Length
- **Target**: Yearly Amount Spent

### Customer Segments Identified
1. Champions
2. Loyal Customers
3. Potential Loyalists
4. New Customers
5. Promising
6. Need Attention
7. About to Sleep
8. At Risk
9. Can't Lose
10. Hibernating

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
openpyxl>=3.0.0
```

## 💻 Usage

### Run E-commerce Prediction Analysis
```python
python ecommerce_prediction.py
```

### Run RFM Customer Analysis
```python
python rfm_customer_analysis.py
```

### Using as a Module
```python
# E-commerce Analysis
from ecommerce_prediction import load_data, train_model, evaluate_model

df = load_data('data/ecommerce-customers.csv')
# ... continue with analysis

# RFM Analysis
from rfm_customer_analysis import CustomerAnalytics

analytics = CustomerAnalytics('data/Online+Retail.xlsx')
analytics.preprocess_data()
analytics.calculate_rfm()
analytics.segment_customers()
analytics.identify_hvc(percentile=80)
analytics.visualize_analysis()
```

## 📈 Visualizations

The project generates comprehensive visualizations including:
- Correlation heatmaps
- Feature relationship plots
- Customer segment distribution
- Revenue analysis charts
- RFM scatter plots
- HVC score distributions

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Data Science**: EDA, statistical analysis, machine learning
- **Python Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Marketing Analytics**: Customer segmentation, predictive modeling
- **Business Intelligence**: Actionable insights generation
- **Data Visualization**: Creating compelling visual stories

## 📝 Key Insights

### E-commerce Findings
- Length of Membership is the strongest predictor of spending
- Mobile app engagement shows higher correlation than website time
- Model achieves high accuracy with minimal features

### RFM Analysis Findings
- Top 20% customers contribute significant revenue share
- Clear segmentation enables targeted marketing strategies
- At-risk customers identified for retention campaigns

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**George Essam**

- Completed: January 6, 2026
- Course: Data Science for Marketing Analytics (7.5 hours)
- Instructor: Eng. Mustafa Othman
- Platform: Udemy

## 🙏 Acknowledgments

- Eng. Mustafa Othman for the comprehensive course
- Udemy platform for educational resources
- Open-source community for the excellent Python libraries

## 📧 Contact

For questions or feedback, please open an issue in this repository.

---

⭐ If you found this project helpful, please consider giving it a star!

**Certificate**: [Udemy Certificate](https://ude.my/UC-0f4dcc15-2ebb-4e80-a538-a7f626288ab4)
