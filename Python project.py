import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/Users/armangupta/Downloads/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")
print(data.head())

print("Information:\n", data.info())
print("Description:\n", data.describe(include='all'))

print("Missing Values:\n", data.isnull().sum())

data['Customer Lifetime Value'] = pd.to_numeric(data['Customer Lifetime Value'], errors='coerce')
data['Income'] = data['Income'].fillna(data['Income'].mean())
data = data.dropna()

data = data.drop_duplicates()
print("Cleaned Data Shape:", data.shape)

print("First 5 Rows:\n", data.head())
print("Last 5 Rows:\n", data.tail())
print("Columns:\n", data.columns)
print("Data Types:\n", data.dtypes)

data.to_csv("cleaned_marketing_dataset.csv", index=False)

top_states = data['State'].value_counts().head(5)
print("Top 5 States:\n", top_states)

top_response = data['Response'].value_counts()
print("Response Counts:\n", top_response)

plt.figure(figsize=(8,5))
state_data = top_states.reset_index()
state_data.columns = ['State', 'Customer_Count']
sns.barplot(data=state_data, x='Customer_Count', y='State', hue='State', palette='Set2', legend=False)
plt.title('Top 5 States with Most Customers')
plt.xlabel('Number of Customers')
plt.ylabel('State')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=data, x='Response', palette='Set1')
plt.title('Response Distribution')
plt.xlabel('Response')
plt.ylabel('Count')
plt.show()

top_sales_channels = data['Sales Channel'].value_counts().head(5)
print("Top Sales Channels:\n", top_sales_channels)

plt.figure(figsize=(6,4))
sns.countplot(data=data, x='Sales Channel', order=top_sales_channels.index, palette='Pastel1')
plt.title('Top Sales Channels')
plt.xlabel('Sales Channel')
plt.ylabel('Count')
plt.show()

top_coverages = data['Coverage'].value_counts().head(5)
plt.figure(figsize=(6,4))
sns.barplot(x=top_coverages.values, y=top_coverages.index, palette='Accent')
plt.title('Top 5 Coverages')
plt.xlabel('Count')
plt.ylabel('Coverage')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data['Customer Lifetime Value'], bins=30, kde=True, color='coral')
plt.title('Distribution of Customer Lifetime Value')
plt.xlabel('Customer Lifetime Value')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=data, x='Response', y='Customer Lifetime Value', palette='coolwarm')
plt.title('Customer Lifetime Value by Response')
plt.xlabel('Response')
plt.ylabel('Customer Lifetime Value')
plt.show()

numeric_cols = ['Customer Lifetime Value', 'Income', 'Monthly Premium Auto', 'Months Since Last Claim']
corr = data[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='Spectral', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

sample_data = data[numeric_cols].sample(n=400, random_state=42)
sns.pairplot(sample_data, diag_kind='kde', corner=True, palette='tab10')
plt.suptitle("Pair Plot of Numeric Features", y=1.02)
plt.show()
