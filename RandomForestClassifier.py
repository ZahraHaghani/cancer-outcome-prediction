import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
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

corr_matrix = data.corr()
target_features = ['T-stage', 'LN', 'LV invasion']
corr_with_targets = corr_matrix[target_features]

plt.figure(figsize=(10, 8))
sns.heatmap(corr_with_targets, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation with Target Variables')
plt.savefig('correlation_with_targets.jpg')
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

forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=1)
multi_target_forest.fit(X_train, y_train)

predictions = multi_target_forest.predict(X_test)

def get_and_plot_metrics(y_true, y_pred, feature_name):
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    metrics_df = pd.DataFrame({
        'Class': np.unique(y_true),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    })
    
    print(f"Metrics for {feature_name}:\n{metrics_df}\n")
    
    metrics_df.set_index('Class', inplace=True)
    ax = metrics_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
    ax.set_title(f'Classification Metrics for {feature_name}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'metrics_{feature_name}.jpg')
    plt.show()

for i, feature_name in enumerate(['T-stage', 'LN', 'LV invasion']):
    get_and_plot_metrics(y_test[feature_name], predictions[:, i], feature_name)
    accuracy = accuracy_score(y_test[feature_name], predictions[:, i])
    print(f"Overall accuracy for {feature_name}: {accuracy:.2f}")

forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

plt.figure(figsize=(12, 8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.savefig('feature_importances.jpg')
plt.show()
