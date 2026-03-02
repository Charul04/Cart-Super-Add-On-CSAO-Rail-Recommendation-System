import numpy as np
import pandas as pd
import joblib
import json

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)


data = pd.read_csv("csao_model_ready.csv")



TARGET   = 'label'
FEATURES = [col for col in data.columns if col != TARGET]

X = data[FEATURES]                                  
y = data[TARGET]                                     

print(f"Dataset shape      : {data.shape}")
print(f"Number of features : {len(FEATURES)}")
print(f"Positive (label=1) : {y.sum():,}  ({y.mean()*100:.1f}%)")
print(f"Negative (label=0) : {(y==0).sum():,}  ({(1-y.mean())*100:.1f}%)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

neg_count        = (y_train == 0).sum()
pos_count        = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count


model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    min_child_weight  = 5,                
    gamma             = 0.1,           
    reg_alpha         = 0.1,            
    reg_lambda        = 1.0,               
    scale_pos_weight  = scale_pos_weight,
    random_state=42,
    n_jobs            = -1,                
    verbosity         = 1,
     early_stopping_rounds=20 
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    
    verbose=50
)

best_round = model.best_iteration
print(f"   Best round        : {best_round}")
print(f"   Best AUC on test  : {model.best_score:.4f}")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


auc       = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
accuracy  = (y_pred == y_test.values).mean()


print(f"\n  AUC Score  : {auc:.4f}")
print(f"  Accuracy   : {accuracy:.4f}")
print(f"  Precision  : {precision:.4f}")
print(f"  Recall     : {recall:.4f} ")
print(f"  F1 Score   : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Rejected (0)', 'Accepted (1)']))


y_true_arr  = y_test.reset_index(drop=True)
y_score_arr = y_pred_proba
def precision_at_k(y_true, y_scores, k):
    """Of the Top-K recommendations, how many did user actually accept?"""
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return y_true.iloc[top_k_idx].sum() / k

def recall_at_k(y_true, y_scores, k):
    """Of all items user would accept, how many are in our Top-K?"""
    top_k_idx     = np.argsort(y_scores)[::-1][:k]
    total_positive = y_true.sum()
    if total_positive == 0:
        return 0.0
    return y_true.iloc[top_k_idx].sum() / total_positive

def ndcg_at_k(y_true, y_scores, k):
    """Ranking quality — rewards putting accepted items higher in the list."""
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    gains     = y_true.iloc[top_k_idx].values
    discounts = np.log2(np.arange(2, k + 2))          
    dcg       = np.sum(gains / discounts)

    ideal_gains = np.sort(y_true.values)[::-1][:k]    # best possible ranking
    idcg        = np.sum(ideal_gains / discounts[:len(ideal_gains)])

    return dcg / idcg if idcg > 0 else 0.0

print(f"{'K':>4} | {'Precision@K':>12} | {'Recall@K':>10} | {'NDCG@K':>8}")
print("-" * 45)
for k in [5, 8, 10]:
    p = precision_at_k(y_true_arr, y_score_arr, k)
    r = recall_at_k(y_true_arr, y_score_arr, k)
    n = ndcg_at_k(y_true_arr, y_score_arr, k)
    print(f"{k:>4} | {p:>12.4f} | {r:>10.4f} | {n:>8.4f}")



feat_imp = pd.DataFrame({
    'feature':    FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print(f"\n{'Rank':>4} | {'Feature':<38} | {'Importance':>10} | Bar")
print("-" * 75)
for i, row in feat_imp.head(15).iterrows():
    bar = "█" * int(row['importance'] * 300)
    print(f"  {i+1:2}. | {row['feature']:<38} | {row['importance']:>10.4f} | {bar}")


# Plot 1 — Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))
top15  = feat_imp.head(15)
colors = ['#e74c3c' if imp > top15['importance'].median() else '#3498db'
          for imp in top15['importance']]
bars = ax.barh(top15['feature'][::-1], top15['importance'][::-1], color=colors[::-1])
for bar, val in zip(bars, top15['importance'][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.set_title('CSAO XGBoost — Top 15 Feature Importances\n(Red = Above Median Importance)',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('csao_feature_importance.png', dpi=150, bbox_inches='tight')

# Plot 2 — Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Rejected', 'Accepted'])
disp.plot(ax=ax, colorbar=False, cmap='Oranges')
ax.set_title('Confusion Matrix — CSAO XGBoost', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('csao_confusion_matrix.png', dpi=150, bbox_inches='tight')

# Plot 3 — ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'XGBoost AUC = {auc:.4f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random Baseline')
ax.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — CSAO XGBoost', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('csao_roc_curve.png', dpi=150, bbox_inches='tight')


joblib.dump(model, "csao_model.joblib")

joblib.dump(FEATURES, "csao_features.joblib")



metadata = {
    "auc"              : round(float(auc), 4),
    "dataset_size"     : int(len(data)),
    "n_features"       : int(len(FEATURES)),
    "best_round"       : int(model.best_iteration),
    "scale_pos_weight" : round(float(scale_pos_weight), 2),
    "features"         : FEATURES,
    "top_features"     : feat_imp.head(10)[['feature', 'importance']]
                                 .assign(importance=lambda df:
                                     df['importance'].round(4))
                                 .to_dict(orient='records'),
}
with open("csao_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


feat_imp.to_csv("csao_feature_importance.csv", index=False)

