import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, f_oneway

# Load the dataset
data = pd.read_csv('/Users/sonuh/software/python-work/sandeep/MLAnomlyDetection/creditcard.csv')
#data = pd.read_csv('/Users/sonuh/software/python-work/sandeep/MLAnomlyDetection/train.csv')
#data = pd.read_csv('/Users/sonuh/software/python-work/sandeep/MLAnomlyDetection/real_estate_data.csv')



# Summary statistics
#print(data.describe())
#print("Median ",data.median())

# Step 2: Identify column types
numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
# Fill missing values in categorical columns
data[categorical_columns] = data[categorical_columns].fillna("Unknown")
# Replace NaN and whitespace-only values with "Unknown"
data[categorical_columns] = data[categorical_columns].map(
    lambda x: "Unknown" if pd.isna(x) or str(x).strip() == "" else x
)

# Step 3: You can inspect unique values for all categorical columns:
if len(categorical_columns) > 0:
    print("Unique values for categorical_columns and count of each catogry, ")
    for col in categorical_columns:
        #print(f"{col}: {data[col].unique()}")
        #Combine smaller categories into a larger group (e.g., "Other") indian, chineese, japaneese, koreans , others.
        data[col] = data[col].apply(lambda x: x if data[col].value_counts()[x] > 20 else "Other")
        print(f"Category '{col}' distribution:")
        print(data[col].value_counts())
    
# Step 4: Compute correlations
# A. Pearson for numerical-numerical correlations
numerical_corr = data[numerical_columns].corr(method='pearson')

# B. Cramér's V for categorical-categorical correlations
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

categorical_corr = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
for col1 in categorical_columns:
    for col2 in categorical_columns:
        categorical_corr.loc[col1, col2] = cramers_v(data[col1], data[col2])

categorical_corr = categorical_corr.astype(float)

# C. ANOVA for numerical-categorical correlations
anova_corr = pd.DataFrame(index=numerical_columns, columns=categorical_columns)
for num_col in numerical_columns:
    for cat_col in categorical_columns:
        unique_vals = data[cat_col].unique()
        groups = [data[data[cat_col] == val][num_col] for val in unique_vals]
        f_stat, _ = f_oneway(*groups)
        anova_corr.loc[num_col, cat_col] = f_stat

anova_corr = anova_corr.astype(float)

# Step 5: Combine all correlation matrices
# Initialize combined matrix with Pearson correlations
combined_corr = pd.DataFrame(index=data.columns, columns=data.columns)

# Fill in numerical-numerical correlations
for col1 in numerical_columns:
    for col2 in numerical_columns:
        combined_corr.loc[col1, col2] = numerical_corr.loc[col1, col2]

# Fill in categorical-categorical correlations
for col1 in categorical_columns:
    for col2 in categorical_columns:
        combined_corr.loc[col1, col2] = categorical_corr.loc[col1, col2]

# Fill in numerical-categorical correlations (from ANOVA results)
for num_col in numerical_columns:
    for cat_col in categorical_columns:
        combined_corr.loc[num_col, cat_col] = anova_corr.loc[num_col, cat_col]
        combined_corr.loc[cat_col, num_col] = anova_corr.loc[num_col, cat_col]  # Symmetry

combined_corr = combined_corr.astype(float)

#Step 6 print the corelation matrix in CSV or numbers
corr_matrix = abs(combined_corr.round(3))
print(corr_matrix.to_csv(index=True)) 
correlation_threshold = 0.1  # Adjust this threshold as needed
target_feature = "Class" # Fraud / Ok transaction
#Low correlation with target
low_correlation_features = corr_matrix[target_feature][abs(corr_matrix[target_feature]) < correlation_threshold].index.tolist()
print(f"Features with low correlation to '{target_feature}' :",low_correlation_features)

#Hight correlation with other features hence one of them can me marked as redundant
high_correlation_threshold = 0.9  # Adjust this threshold as needed
highly_correlated_features = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > high_correlation_threshold:
            highly_correlated_features.append((corr_matrix.columns[i], corr_matrix.columns[j]))
print("Potentially redundant feature pairs:",highly_correlated_features)


# Step 6: Generate heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(combined_corr, cmap='coolwarm', annot=False, vmin=-1, vmax=1)
plt.title("Combined Correlation Matrix")
plt.show()


#Feature reduction (that are not corelated with the end goal -Grey color, 
# or when two feature are highly corelared to each other (other than to goal Too red/blue other than diagonal) making oneof them redundant),
# composite feature,


#One hot-encding, label to convert the categorical features into numericals so that the model can work on them