# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class CustomerAnalytics:
    """Class for RFM and HVC analysis"""
    
    def __init__(self, filepath):
        """Initialize and load data"""
        self.df = pd.read_excel(filepath)
        self.dfc = self.df.copy()
        self.rfm = None
        self.crfm = None
        
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        print(f"\nOriginal shape: {self.dfc.shape}")
        print(f"\nMissing values:\n{self.dfc.isna().sum()}")
        
        # Remove missing values
        self.dfc.dropna(inplace=True)
        print(f"\nShape after removing NA: {self.dfc.shape}")
        
        # Remove cancelled orders
        initial_size = len(self.dfc)
        self.dfc = self.dfc[~self.dfc['InvoiceNo'].str.contains('C', na=False)]
        print(f"Removed {initial_size - len(self.dfc)} cancelled orders")
        
        # Calculate total price
        self.dfc['TotalPrice'] = self.dfc['Quantity'] * self.dfc['UnitPrice']
        
        # Remove negative quantities and prices
        self.dfc = self.dfc[(self.dfc['Quantity'] > 0) & (self.dfc['TotalPrice'] > 0)]
        print(f"\nFinal shape: {self.dfc.shape}")
        
        return self.dfc
    
    def calculate_rfm(self):
        """Calculate RFM metrics"""
        print("\n" + "=" * 80)
        print("RFM ANALYSIS")
        print("=" * 80)
        
        NOW = self.dfc['InvoiceDate'].max() + dt.timedelta(days=1)
        print(f"\nAnalysis reference date: {NOW.date()}")
        
        self.rfm = self.dfc.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (NOW - x.max()).days,
            'InvoiceNo': lambda x: x.nunique(),
            'TotalPrice': lambda x: x.sum()
        })
        
        self.crfm = self.rfm.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalPrice': 'Monetary'
        })
        
        # Filter positive monetary values
        self.crfm = self.crfm[self.crfm['Monetary'] > 0]
        
        print(f"\nTotal customers analyzed: {len(self.crfm)}")
        print(f"\nRFM Summary Statistics:")
        print(self.crfm.describe().round(2))
        
        return self.crfm
    
    def segment_customers(self):
        """Create RFM segments"""
        print("\n" + "=" * 80)
        print("CUSTOMER SEGMENTATION")
        print("=" * 80)
        
        # Calculate RFM scores
        self.crfm['R_Score'] = pd.qcut(self.crfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        self.crfm['F_Score'] = pd.qcut(self.crfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        self.crfm['M_Score'] = pd.qcut(self.crfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        # Create RFM segment
        self.crfm['RFM_Segment'] = self.crfm['R_Score'].astype(str) + self.crfm['F_Score'].astype(str)
        
        # Map segments to customer categories
        seg_map = {
            r'[1-2][1-2]': 'Hibernating',
            r'[1-2][3-4]': 'At_Risk',
            r'[1-2]5': 'Cant_Loose',
            r'3[1-2]': 'About_to_Sleep',
            r'33': 'Need_Attention',
            r'[3-4][4-5]': 'Loyal_Customers',
            r'41': 'Promising',
            r'51': 'New_Customers',
            r'[4-5][2-3]': 'Potential_Loyalists',
            r'5[4-5]': 'Champions'
        }
        
        self.crfm['Segment'] = self.crfm['RFM_Segment'].replace(seg_map, regex=True)
        
        # Display segment distribution
        print("\nSegment Distribution:")
        print(self.crfm['Segment'].value_counts().sort_values(ascending=False))
        
        # Segment characteristics
        print("\nSegment Characteristics:")
        segment_summary = self.crfm.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum', 'count']
        }).round(2)
        segment_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue', 'Customer_Count']
        print(segment_summary.sort_values('Total_Revenue', ascending=False))
        
        return self.crfm
    
    def identify_hvc(self, percentile=80):
        """
        Identify High-Value Customers (HVC)
        HVC are defined as customers in the top percentile by monetary value
        """
        print("\n" + "=" * 80)
        print("HIGH-VALUE CUSTOMER (HVC) ANALYSIS")
        print("=" * 80)
        
        # Define HVC threshold
        hvc_threshold = self.crfm['Monetary'].quantile(percentile / 100)
        print(f"\nHVC Threshold (Top {100-percentile}%): ${hvc_threshold:,.2f}")
        
        # Identify HVCs
        self.crfm['Customer_Type'] = self.crfm['Monetary'].apply(
            lambda x: 'High-Value Customer' if x >= hvc_threshold else 'Regular Customer'
        )
        
        # HVC statistics
        hvc_count = (self.crfm['Customer_Type'] == 'High-Value Customer').sum()
        hvc_percentage = (hvc_count / len(self.crfm)) * 100
        
        print(f"\nTotal HVCs: {hvc_count} ({hvc_percentage:.1f}% of customer base)")
        
        # Revenue contribution
        hvc_revenue = self.crfm[self.crfm['Customer_Type'] == 'High-Value Customer']['Monetary'].sum()
        total_revenue = self.crfm['Monetary'].sum()
        hvc_revenue_pct = (hvc_revenue / total_revenue) * 100
        
        print(f"HVC Revenue Contribution: ${hvc_revenue:,.2f} ({hvc_revenue_pct:.1f}% of total)")
        
        # HVC comparison
        print("\nComparison: HVC vs Regular Customers")
        comparison = self.crfm.groupby('Customer_Type').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum', 'count']
        }).round(2)
        comparison.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue', 'Count']
        print(comparison)
        
        # HVC segment distribution
        print("\nHVC Distribution Across Segments:")
        hvc_segments = pd.crosstab(
            self.crfm['Segment'], 
            self.crfm['Customer_Type'], 
            margins=True
        )
        print(hvc_segments)
        
        # Calculate HVC score (composite metric)
        self.crfm['HVC_Score'] = (
            self.crfm['R_Score'].astype(int) * 0.3 +
            self.crfm['F_Score'].astype(int) * 0.3 +
            self.crfm['M_Score'].astype(int) * 0.4
        )
        
        # Identify top HVCs
        print("\nTop 10 High-Value Customers:")
        top_hvc = self.crfm.nlargest(10, 'Monetary')[['Recency', 'Frequency', 'Monetary', 'Segment', 'HVC_Score']]
        print(top_hvc)
        
        return self.crfm
    
    def visualize_analysis(self):
        """Create visualizations for RFM and HVC analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Segment distribution
        segment_counts = self.crfm['Segment'].value_counts()
        axes[0, 0].bar(range(len(segment_counts)), segment_counts.values, color='steelblue')
        axes[0, 0].set_xticks(range(len(segment_counts)))
        axes[0, 0].set_xticklabels(segment_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('Customer Segment Distribution')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # 2. Revenue by segment
        segment_revenue = self.crfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
        axes[0, 1].barh(range(len(segment_revenue)), segment_revenue.values, color='coral')
        axes[0, 1].set_yticks(range(len(segment_revenue)))
        axes[0, 1].set_yticklabels(segment_revenue.index)
        axes[0, 1].set_title('Total Revenue by Segment')
        axes[0, 1].set_xlabel('Revenue ($)')
        
        # 3. HVC vs Regular distribution
        customer_type_counts = self.crfm['Customer_Type'].value_counts()
        colors = ['#ff9999', '#66b3ff']
        axes[0, 2].pie(customer_type_counts.values, labels=customer_type_counts.index, 
                       autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 2].set_title('Customer Type Distribution')
        
        # 4. RFM distribution
        axes[1, 0].scatter(self.crfm['Recency'], self.crfm['Monetary'], 
                          c=self.crfm['Frequency'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Recency (days)')
        axes[1, 0].set_ylabel('Monetary ($)')
        axes[1, 0].set_title('Recency vs Monetary (colored by Frequency)')
        plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Frequency')
        
        # 5. HVC Score distribution
        axes[1, 1].hist(self.crfm['HVC_Score'], bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('HVC Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('HVC Score Distribution')
        axes[1, 1].axvline(self.crfm['HVC_Score'].mean(), color='red', 
                           linestyle='--', label=f'Mean: {self.crfm["HVC_Score"].mean():.2f}')
        axes[1, 1].legend()
        
        # 6. Customer Type by Segment
        ct_segment = pd.crosstab(self.crfm['Segment'], self.crfm['Customer_Type'])
        ct_segment.plot(kind='bar', stacked=True, ax=axes[1, 2], color=['lightcoral', 'lightblue'])
        axes[1, 2].set_title('Customer Type by Segment')
        axes[1, 2].set_xlabel('Segment')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend(title='Customer Type')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def generate_insights(self):
        """Generate actionable insights"""
        print("\n" + "=" * 80)
        print("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)
        
        total_customers = len(self.crfm)
        hvc_count = (self.crfm['Customer_Type'] == 'High-Value Customer').sum()
        
        print("\n1. CUSTOMER BASE OVERVIEW:")
        print(f"   - Total active customers: {total_customers:,}")
        print(f"   - High-value customers: {hvc_count:,} ({(hvc_count/total_customers)*100:.1f}%)")
        
        print("\n2. PRIORITY SEGMENTS:")
        top_segments = self.crfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False).head(3)
        for i, (segment, revenue) in enumerate(top_segments.items(), 1):
            count = (self.crfm['Segment'] == segment).sum()
            print(f"   {i}. {segment}: {count} customers, ${revenue:,.2f} revenue")
        
        print("\n3. AT-RISK CUSTOMERS:")
        at_risk = self.crfm[self.crfm['Segment'].isin(['At_Risk', 'Cant_Loose', 'Hibernating'])]
        print(f"   - {len(at_risk):,} customers need immediate attention")
        print(f"   - Potential revenue loss: ${at_risk['Monetary'].sum():,.2f}")
        
        print("\n4. RECOMMENDATIONS:")
        print("   - Champions & Loyal: Reward programs and exclusive offers")
        print("   - At Risk & Can't Loose: Win-back campaigns with special incentives")
        print("   - New Customers: Onboarding programs to increase engagement")
        print("   - Potential Loyalists: Cross-sell and upsell opportunities")
        print("   - HVCs: Personalized service and VIP treatment")


# Usage example
if __name__ == "__main__":
    # Initialize analysis
    filepath = r'D:\VS_code\VS_code_WorkSpace\python_projects\real\Online+Retail.xlsx'
    analytics = CustomerAnalytics(filepath)
    
    # Run complete analysis pipeline
    analytics.preprocess_data()
    analytics.calculate_rfm()
    analytics.segment_customers()
    analytics.identify_hvc(percentile=80)  # Top 20% are HVCs
    analytics.visualize_analysis()
    analytics.generate_insights()
    
    # Export results
    output_file = 'customer_analysis_results.csv'
    analytics.crfm.to_csv(output_file)
    print(f"\n\nResults exported to: {output_file}")