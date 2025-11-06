# ==========================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM - ADVANCED GUI
# ==========================================================

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

# ===========================
# LOAD DATASET
# ===========================
data = pd.read_csv("D:\\project 1\\Dataset\\Student_Performance.csv")

# Process dataset
data['Result'] = data['Performance Index'].apply(lambda x: 'Pass' if x >= 33 else 'Fail')
le = LabelEncoder()
data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
          'Sleep Hours', 'Sample Question Papers Practiced']]
y_class = data['Result']
y_reg = data['Performance Index']

# Split + scale
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# TRAIN MODELS
# ===========================
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Naive Bayes": GaussianNB()
}
best_model_name, best_acc = None, 0
for name, model in models.items():
    model.fit(X_train_scaled, y_train_class)
    acc = model.score(X_test_scaled, y_test_class)
    if acc > best_acc:
        best_model_name, best_acc, best_model = name, acc, model

# Regression model for percentage
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_scaled, data.loc[X_train.index, 'Performance Index'])

# ===========================
# GUI START
# ===========================
root = tk.Tk()
root.title("🎓 Student Performance Prediction System")
root.attributes('-fullscreen', True)
root.configure(bg="#e8f0fe")

def exit_fullscreen(event=None): root.attributes('-fullscreen', False)
def close_app(): root.destroy()
root.bind("<Escape>", exit_fullscreen)

# -------------------- HEADER --------------------
header = tk.Frame(root, bg="#003366", height=90)
header.pack(fill='x')
tk.Label(header, text="Student Performance Prediction System",
         font=("Helvetica", 26, "bold"), bg="#003366", fg="white").pack(pady=25)

# -------------------- TAB CONTROL --------------------
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# ==========================================================
# TAB A: PREDICTION
# ==========================================================
tab_predict = tk.Frame(notebook, bg="#e8f0fe")
notebook.add(tab_predict, text="🔮 Prediction")

def create_label(tab, text):
    return tk.Label(tab, text=text, font=("Arial", 14, "bold"), bg="#e8f0fe", fg="#333")

def create_entry(tab):
    entry = tk.Entry(tab, font=("Arial", 14), bd=1, relief="solid", width=35)
    entry.configure(highlightthickness=1, highlightbackground="#b0c4de", highlightcolor="#003366")
    return entry

# Input fields
create_label(tab_predict, "Hours Studied:").pack(pady=6, anchor='w')
hours_entry = create_entry(tab_predict); hours_entry.pack(pady=5)

create_label(tab_predict, "Previous Scores:").pack(pady=6, anchor='w')
prev_score_entry = create_entry(tab_predict); prev_score_entry.pack(pady=5)

create_label(tab_predict, "Extracurricular Activities (Yes/No):").pack(pady=6, anchor='w')
extra_var = tk.StringVar()
extra_dropdown = tk.OptionMenu(tab_predict, extra_var, "Yes", "No")
extra_dropdown.config(font=("Arial", 14), bg="white", width=33)
extra_dropdown.pack(pady=5)

create_label(tab_predict, "Sleep Hours:").pack(pady=6, anchor='w')
sleep_entry = create_entry(tab_predict); sleep_entry.pack(pady=5)

create_label(tab_predict, "Sample Question Papers Practiced:").pack(pady=6, anchor='w')
papers_entry = create_entry(tab_predict); papers_entry.pack(pady=5)

# Progress Bar
progress_label = tk.Label(tab_predict, text="", bg="#e8f0fe", font=("Arial", 14))
progress_label.pack(pady=5)
progress = ttk.Progressbar(tab_predict, orient="horizontal", length=400, mode="determinate")
progress.pack(pady=5)

# Prediction function
def predict_result():
    try:
        hours = float(hours_entry.get())
        previous = float(prev_score_entry.get())
        extra = extra_var.get()
        sleep = float(sleep_entry.get())
        papers_input = papers_entry.get().strip().lower()
        papers = 0 if papers_input in ['no', 'none', ''] else int(papers_input)

        if extra not in ['Yes', 'No']:
            messagebox.showerror("Error", "Extracurricular Activities must be Yes or No.")
            return

        extra_encoded = le.transform([extra])[0]
        input_df = pd.DataFrame([[hours, previous, extra_encoded, sleep, papers]],
                                columns=X.columns)
        input_scaled = scaler.transform(input_df)

        result = best_model.predict(input_scaled)[0]
        percent = np.clip(regressor.predict(input_scaled)[0], 0, 100)

        progress['value'] = percent
        progress_label.config(text=f"Predicted Percentage: {percent:.2f}%")

        messagebox.showinfo("Prediction Result",
                            f"🎓 Result: {result}\n📈 Predicted Percentage: {percent:.2f}%\n\n"
                            f"Model: {best_model_name}\nAccuracy: {round(best_acc*100,2)}%")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Buttons
def on_enter(e): predict_btn.config(bg="#2e8b57")
def on_leave(e): predict_btn.config(bg="#4CAF50")

predict_btn = tk.Button(tab_predict, text="Predict Result", font=("Arial", 16, "bold"),
                        bg="#4CAF50", fg="white", command=predict_result, width=20)
predict_btn.pack(pady=25)
predict_btn.bind("<Enter>", on_enter)
predict_btn.bind("<Leave>", on_leave)

# ==========================================================
# TAB B: DATA VISUALIZATION DASHBOARD
# ==========================================================
tab_visual = tk.Frame(notebook, bg="#e8f0fe")
notebook.add(tab_visual, text="📊 Data Visualization")

def show_correlation():
    plt.figure(figsize=(8,6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def show_feature_importance():
    importances = best_model.feature_importances_
    plt.figure(figsize=(7,5))
    plt.barh(X.columns, importances, color='skyblue')
    plt.title("Feature Importance in Prediction")
    plt.xlabel("Importance")
    plt.show()

def show_pass_fail_ratio():
    counts = data['Result'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=["#4CAF50","#f44336"])
    plt.title("Pass vs Fail Distribution")
    plt.show()

tk.Button(tab_visual, text="📈 Correlation Heatmap", command=show_correlation,
          font=("Arial",14), bg="#1976D2", fg="white", width=30).pack(pady=20)
tk.Button(tab_visual, text="🌟 Feature Importance", command=show_feature_importance,
          font=("Arial",14), bg="#388E3C", fg="white", width=30).pack(pady=20)
tk.Button(tab_visual, text="🎯 Pass/Fail Ratio", command=show_pass_fail_ratio,
          font=("Arial",14), bg="#E64A19", fg="white", width=30).pack(pady=20)

# ==========================================================
# TAB C: ABOUT PROJECT
# ==========================================================
tab_about = tk.Frame(notebook, bg="#f4f9ff")
notebook.add(tab_about, text="ℹ️ About Project")

about_text = (
    "🎓 STUDENT PERFORMANCE PREDICTION SYSTEM\n\n"
    "This system predicts a student's academic performance and approximate percentage\n"
    "based on study habits, previous scores, sleep hours, and practice frequency.\n\n"
    "🧠 Algorithms Used:\n"
    "• Random Forest Classifier\n• Support Vector Machine (SVM)\n• Naive Bayes\n\n"
    "💡 Features:\n"
    "• Pass/Fail Prediction\n• Percentage Estimation\n• Visualization Dashboard\n"
    "• Full-Screen Interactive GUI\n\n"
    "👥 Developed By:\n"
    "• Nikhil Mukati\n• Nikhil Rathore\n• Nihal Dubey\n\n"
    "📘 Tools & Technologies:\n"
    "Python, Tkinter, Scikit-learn, Matplotlib, Seaborn\n\n"
    "🏫 Objective:\n"
    "To assist institutions in identifying at-risk students and improving academic performance."
)

tk.Label(tab_about, text=about_text, justify="left",
         font=("Arial", 14), bg="#f4f9ff", fg="#333", padx=30, pady=20).pack(anchor="w")

# ==========================================================
# FOOTER
# ==========================================================
footer = tk.Frame(root, bg="#003366", height=60)
footer.pack(fill='x', side='bottom')
tk.Label(footer, text=f"Best Model: {best_model_name} ({round(best_acc*100,2)}%)",
         font=("Arial", 12), bg="#003366", fg="white").pack(side='left', padx=30)
tk.Button(footer, text="Exit", font=("Arial", 12, "bold"),
          bg="red", fg="white", command=close_app, width=10).pack(side='right', padx=30, pady=10)

root.mainloop()
