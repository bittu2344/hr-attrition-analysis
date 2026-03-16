# ============================================================
# PROJECT 3: HR Employee Attrition Analysis
# Author: Bittu Kumar | Data Analyst
# Tools: Python | Pandas | Seaborn | Matplotlib | Excel
# Dataset: IBM HR Analytics Employee Attrition (Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  HR EMPLOYEE ATTRITION ANALYSIS")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
# Download: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Save as 'WA_Fn-UseC_-HR-Employee-Attrition.csv' in this folder

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"\n✅ Data Loaded: {df.shape[0]:,} employees, {df.shape[1]} columns")
print(df.head(3))


# ─────────────────────────────────────────
# STEP 2: DATA CLEANING & OVERVIEW
# ─────────────────────────────────────────
print("\n📋 Data Types & Missing Values:")
print(df.dtypes[df.dtypes == 'object'].index.tolist(), " ← Categorical")
print(f"Missing Values: {df.isnull().sum().sum()}")

# Remove useless columns
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=drop_cols, inplace=True)

# Encode Attrition
df['Attrition_Flag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

attrition_rate = df['Attrition_Flag'].mean() * 100
print(f"\n📊 Overall Attrition Rate: {attrition_rate:.1f}%")


# ─────────────────────────────────────────
# STEP 3: EDA - 15+ VISUALIZATIONS
# ─────────────────────────────────────────
sns.set_style('whitegrid')
palette = {'Yes': '#e74c3c', 'No': '#3498db'}

# ── Figure 1: Overview Charts ──────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle('HR Attrition Analysis — Overview', fontsize=16, fontweight='bold')

# 1. Attrition Pie Chart
att_counts = df['Attrition'].value_counts()
axes[0, 0].pie(att_counts, labels=['Stayed', 'Left'],
               autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
axes[0, 0].set_title('Attrition Distribution')

# 2. Age Distribution by Attrition
sns.histplot(data=df, x='Age', hue='Attrition', bins=20,
             palette=palette, ax=axes[0, 1], kde=True)
axes[0, 1].set_title('Age Distribution by Attrition')
axes[0, 1].axvspan(25, 35, alpha=0.1, color='orange', label='High Risk Zone')
axes[0, 1].legend()

# 3. Overtime vs Attrition
ot = df.groupby(['OverTime', 'Attrition']).size().unstack()
ot.plot(kind='bar', ax=axes[0, 2], color=['#3498db', '#e74c3c'],
        edgecolor='white', rot=0)
axes[0, 2].set_title('Overtime vs Attrition')
axes[0, 2].set_xlabel('Overtime')
axes[0, 2].set_ylabel('Count')
axes[0, 2].legend(['No Attrition', 'Attrition'])

# 4. Department Attrition
dept_att = df.groupby('Department')['Attrition_Flag'].mean() * 100
axes[1, 0].bar(dept_att.index, dept_att.values, color=['#f39c12', '#9b59b6', '#1abc9c'])
axes[1, 0].set_title('Attrition Rate by Department (%)')
axes[1, 0].set_ylabel('Attrition Rate (%)')
plt.setp(axes[1, 0].get_xticklabels(), rotation=15, ha='right')
for i, v in enumerate(dept_att.values):
    axes[1, 0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)

# 5. Job Satisfaction vs Attrition
sns.boxplot(data=df, x='Attrition', y='JobSatisfaction',
            palette=palette, ax=axes[1, 1])
axes[1, 1].set_title('Job Satisfaction vs Attrition')

# 6. Monthly Income by Attrition
sns.violinplot(data=df, x='Attrition', y='MonthlyIncome',
               palette=palette, ax=axes[1, 2])
axes[1, 2].set_title('Monthly Income vs Attrition')
axes[1, 2].set_ylabel('Monthly Income (₹)')

plt.tight_layout()
plt.savefig('attrition_overview.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 2: Deep Dive Charts ─────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('HR Attrition Analysis — Deep Dive', fontsize=16, fontweight='bold')

# 7. Years at Company
sns.histplot(data=df, x='YearsAtCompany', hue='Attrition',
             bins=15, palette=palette, ax=axes2[0, 0], kde=True)
axes2[0, 0].set_title('Years at Company vs Attrition')

# 8. Job Role Attrition Rate
role_att = (df.groupby('JobRole')['Attrition_Flag']
              .mean() * 100).sort_values(ascending=False)
axes2[0, 1].barh(role_att.index, role_att.values, color='#e67e22')
axes2[0, 1].set_title('Attrition Rate by Job Role (%)')
axes2[0, 1].set_xlabel('Attrition Rate (%)')

# 9. Distance from Home
sns.boxplot(data=df, x='Attrition', y='DistanceFromHome',
            palette=palette, ax=axes2[0, 2])
axes2[0, 2].set_title('Distance from Home vs Attrition')

# 10. Work-Life Balance
wlb = df.groupby(['WorkLifeBalance', 'Attrition']).size().unstack(fill_value=0)
wlb.plot(kind='bar', ax=axes2[1, 0], color=['#3498db', '#e74c3c'],
         edgecolor='white', rot=0)
axes2[1, 0].set_title('Work-Life Balance vs Attrition')
axes2[1, 0].set_xlabel('Work-Life Balance (1=Bad, 4=Best)')

# 11. Age 25–35 + Overtime: HIGH RISK GROUP
high_risk = df[(df['Age'].between(25, 35)) & (df['OverTime'] == 'Yes')]
hr_rate = high_risk['Attrition_Flag'].mean() * 100
normal_rate = df['Attrition_Flag'].mean() * 100
axes2[1, 1].bar(['Overall', 'Age 25-35\n+ Overtime'],
                [normal_rate, hr_rate],
                color=['#3498db', '#e74c3c'])
axes2[1, 1].set_title('⚠️ High Risk Group Attrition')
axes2[1, 1].set_ylabel('Attrition Rate (%)')
for i, v in enumerate([normal_rate, hr_rate]):
    axes2[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center',
                     fontsize=12, fontweight='bold')

# 12. Correlation Heatmap
num_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction',
            'DistanceFromHome', 'WorkLifeBalance', 'Attrition_Flag']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes2[1, 2], linewidths=0.5)
axes2[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('attrition_deep_dive.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 3: Summary Insights ──────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS FROM HR ATTRITION ANALYSIS")
print("=" * 55)
print(f"  📊 Overall Attrition Rate   : {attrition_rate:.1f}%")
print(f"  ⚠️  High-Risk Group (25-35+OT): {hr_rate:.1f}% attrition")
print(f"     → {hr_rate - normal_rate:.1f}% HIGHER than average")

top_dept = dept_att.idxmax()
print(f"  🏢 Highest Attrition Dept   : {top_dept} ({dept_att.max():.1f}%)")
print(f"  💰 Avg Income (Left)        : "
      f"₹{df[df['Attrition']=='Yes']['MonthlyIncome'].mean():,.0f}")
print(f"  💰 Avg Income (Stayed)      : "
      f"₹{df[df['Attrition']=='No']['MonthlyIncome'].mean():,.0f}")
print("\n✅ Analysis Complete! Charts saved as PNG files.")
