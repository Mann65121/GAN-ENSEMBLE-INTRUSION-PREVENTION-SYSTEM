import pandas as pd
import os

os.makedirs("results", exist_ok=True)

data = {
    "Class": [
        "Normal", "Class_1", "Class_2", "Class_3", "Class_4",
        "Class_5", "Class_6", "Class_7", "Class_8", "Class_9",
        "Accuracy", "Macro Avg", "Weighted Avg"
    ],
    "Precision": [
        0.5167, 0.2982, 0.4266, 0.5997, 0.6762,
        0.9891, 0.9085, 0.8065, 0.5475, 0.2778,
        "", 0.6047, 0.8035
    ],
    "Recall": [
        0.0772, 0.0243, 0.0510, 0.8871, 0.5958,
        0.9749, 0.9453, 0.7104, 0.2671, 0.0962,
        0.8177, 0.4629, 0.8177
    ],
    "F1-Score": [
        0.1343, 0.0450, 0.0910, 0.7156, 0.6335,
        0.9820, 0.9265, 0.7554, 0.3591, 0.1429,
        "", 0.4785, 0.7928
    ],
    "Support": [
        803, 699, 4906, 13358, 7274,
        17661, 27900, 4196, 453, 52,
        77302, 77302, 77302
    ]
}

df = pd.DataFrame(data)
df.to_csv("results/final_project_result_unsw_multiclass.csv", index=False)

print("✅ Final project result saved in ONE file:")
print("➡ results/final_project_result_unsw_multiclass.csv")
