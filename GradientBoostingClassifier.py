import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

data_path = 'updated_transformed_data.xlsx'
data = pd.read_excel(data_path)
sns.set_style("darkgrid")
fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(15, 40))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    sns.histplot(data[col], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}', fontsize=15)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
plt.tight_layout()
plt.savefig('feature_distributions.jpg')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.jpg')
plt.show()

subset_columns = ['Age', 'T-stage', 'LN', 'LV invasion', 'MAF (%)']
sns.pairplot(data[subset_columns], hue='LV invasion')
plt.savefig('pairplot.jpg')
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

print("Variance explained by each component:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

sns.pairplot(pca_df)
plt.savefig('pca_results.jpg')
plt.show()

X = data.drop(labels=['T-stage', 'LN', 'LV invasion'], axis=1)
y = data[['T-stage', 'LN', 'LV invasion']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {}
for target in y.columns:
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm.fit(X_train, y_train[target])
    models[target] = gbm
    predictions = gbm.predict(X_test)
    
    accuracy = accuracy_score(y_test[target], predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test[target], predictions, average=None, zero_division=0)
    
    metrics_df = pd.DataFrame({
        'Class': np.unique(y_test[target]),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })
    
    print(f"Metrics for {target}:\n{metrics_df}\n")
    print(f"Overall accuracy for {target}: {accuracy:.2f}")
    
    metrics_df.set_index('Class', inplace=True)
    ax = metrics_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
    ax.set_title(f'Classification Metrics for {target}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'metrics_{target}.jpg')
    plt.show()

    importances = gbm.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"Feature ranking for {target}:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]})")
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importances for {target}")
    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.savefig(f'feature_importances_{target}.jpg')
    plt.show()

