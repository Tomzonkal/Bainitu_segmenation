import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# === 1. Load your trained RandomForest model ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# === 2. Compute structural stats for each decision tree ===
tree_stats = []
for i, estimator in enumerate(model.estimators_):
    depth = estimator.tree_.max_depth
    n_nodes = estimator.tree_.node_count
    tree_stats.append((i, depth, n_nodes))

df = pd.DataFrame(tree_stats, columns=["Tree_Index", "Depth", "Num_Nodes"])
df_sorted = df.sort_values(by="Depth", ascending=False)

print("\n=== Random Forest Tree Summary (Structure Only) ===")
print(df_sorted.to_string(index=False))

# === 3. Optionally show top 5 largest trees ===
print("\nTop 5 largest trees by depth:")
print(df_sorted.head(5).to_string(index=False))

# === 4. Let user choose a tree to visualize ===
while True:
    try:
        tree_id = int(input(f"\nEnter Tree_Index to visualize (0–{len(model.estimators_) - 1}): "))
        if 0 <= tree_id < len(model.estimators_):
            break
        else:
            print("Out of range. Try again.")
    except ValueError:
        print("Please enter a valid integer.")

# === 5. Visualize the chosen tree ===
chosen_tree = model.estimators_[tree_id]
print(f"\nVisualizing Tree #{tree_id} (Depth={chosen_tree.tree_.max_depth}, Nodes={chosen_tree.tree_.node_count})")

# Handle missing feature names
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

plt.figure(figsize=(25, 15))
tree.plot_tree(
    chosen_tree,
    filled=True,
    feature_names=feature_names,
    class_names=model.classes_.astype(str)
)
plt.title(f"Decision Tree #{tree_id} from Random Forest")
plt.show()
