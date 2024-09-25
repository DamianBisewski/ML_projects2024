from ISLP import load_data
data = load_data('OJ')
data['Store7'] = data['Store7'].map({'No': 0, 'Yes': 1})
import pandas as pd
from sklearn.model_selection import train_test_split

# Split the data into training (800) and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

# Separate features and target variable from training and test sets
X_train = train_data.drop(columns=['Purchase'])
y_train = train_data['Purchase']
X_test = test_data.drop(columns=['Purchase'])
y_test = test_data['Purchase']

# Fit a decision tree to the training data
tree_model = DecisionTreeClassifier(random_state=123)
tree_model.fit(X_train, y_train)

# Predict on the training set
train_pred = tree_model.predict(X_train)

# Calculate training error rate
train_error_rate = 1 - accuracy_score(y_train, train_pred)
train_error_rate
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(tree_model, filled=True, feature_names=X_train.columns, class_names=tree_model.classes_, rounded=True)
plt.show()

# Number of terminal nodes
n_terminal_nodes = tree_model.get_n_leaves()
n_terminal_nodes
# Produce a text summary of the tree
tree_text = export_text(tree_model, feature_names=list(X_train.columns))
print(tree_text)
# Predict on the test set
test_pred = tree_model.predict(X_test)

# Confusion matrix
test_confusion = confusion_matrix(y_test, test_pred)
print(test_confusion)

# Test error rate
test_error_rate = 1 - accuracy_score(y_test, test_pred)
test_error_rate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Assume X_train and y_train are already defined
# Cross-validation to determine optimal tree size
path = tree_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

cv_scores = []
alpha_scores = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=123, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    mean_score = scores.mean()
    cv_scores.append(mean_score)
    alpha_scores.append((mean_score, ccp_alpha))
# Find the minimum cv_score and the corresponding ccp_alpha
min_cv_score, min_ccp_alpha = min(alpha_scores, key=lambda x: x[0])

# Print the minimum cv_score and the corresponding ccp_alpha
print(f"Minimum CV Score: {min_cv_score}")
print(f"Corresponding ccp_alpha: {min_ccp_alpha}")

# Optimal alpha based on max cv_score (if needed)
optimal_ccp_alpha = ccp_alphas[cv_scores.index(max(cv_scores))]
print(f"Optimal ccp_alpha (max CV score): {optimal_ccp_alpha}")
# Plotting cross-validated error rate
plt.figure()
plt.plot(ccp_alphas, [1 - score for score in cv_scores], marker='o')
plt.xlabel('Tree size (alpha)')
plt.ylabel('Cross-validated classification error rate')
plt.show()
# Prune the tree based on the optimal alpha value
pruned_tree = DecisionTreeClassifier(random_state=123, ccp_alpha=optimal_ccp_alpha)
pruned_tree.fit(X_train, y_train)

# Plot the pruned tree
plt.figure(figsize=(20,10))
plot_tree(pruned_tree, filled=True, feature_names=X_train.columns, class_names=pruned_tree.classes_, rounded=True)
plt.show()
# Predict on the training set using the pruned tree
train_pred_pruned = pruned_tree.predict(X_train)

# Calculate training error rate for pruned tree
train_error_rate_pruned = 1 - accuracy_score(y_train, train_pred_pruned)
train_error_rate_pruned
# Predict on the test set using the pruned tree
test_pred_pruned = pruned_tree.predict(X_test)

# Calculate test error rate for pruned tree
test_error_rate_pruned = 1 - accuracy_score(y_test, test_pred_pruned)
test_error_rate_pruned
print(f"Training error rate (unpruned): {train_error_rate}")
print(f"Training error rate (pruned): {train_error_rate_pruned}")
print(f"Test error rate (unpruned): {test_error_rate}")
print(f"Test error rate (pruned): {test_error_rate_pruned}")