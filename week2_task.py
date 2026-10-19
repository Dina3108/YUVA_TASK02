# Week 2 - Visualizing Data and Explaining the Patterns
# These charts use the same diabetes dataset as Week 1.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

output_folder = Path("week2_outputs")
output_folder.mkdir(exist_ok=True)

data = load_diabetes(as_frame=True).frame
sns.set_theme(style="whitegrid")

# Chart 1: distribution of the target values.
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="target", bins=25, kde=True)
plt.title("Distribution of Target Values")
plt.xlabel("Target value")
plt.tight_layout()
plt.savefig(output_folder / "target_distribution.png", dpi=150)
plt.close()

# Chart 2: see how the BMI feature and target move together.
plt.figure(figsize=(8, 5))
sns.regplot(data=data, x="bmi", y="target",
            scatter_kws={"alpha": 0.45})
plt.title("BMI Feature Compared with Target")
plt.tight_layout()
plt.savefig(output_folder / "bmi_target_scatter.png", dpi=150)
plt.close()

# Chart 3: a heatmap gives a quick overview of correlations.
plt.figure(figsize=(9, 7))
sns.heatmap(data.corr(numeric_only=True), cmap="vlag", center=0)
plt.title("Correlation Overview")
plt.tight_layout()
plt.savefig(output_folder / "correlation_heatmap.png", dpi=150)
plt.close()

# Chart 4: split the BMI feature into two groups for a simple comparison.
data["BMI group"] = pd.cut(
    data["bmi"], bins=2, labels=["Lower BMI feature", "Higher BMI feature"]
)
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="BMI group", y="target")
plt.title("Target Values Across BMI Feature Groups")
plt.tight_layout()
plt.savefig(output_folder / "target_by_bmi_group.png", dpi=150)
plt.close()

# Chart 5: sort the correlations so the stronger ones are easier to compare.
correlations = data.corr(numeric_only=True)["target"].drop("target").sort_values()
plt.figure(figsize=(8, 5))
correlations.plot(kind="barh")
plt.title("Feature Correlation with the Target")
plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig(output_folder / "feature_target_correlations.png", dpi=150)
plt.close()

print("All five charts have been saved in", output_folder)
