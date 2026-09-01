# 🧠 The Glossarium: Big Data & Kaggle Meta

### 📊 Data Encoding & Scaling
* **Nominal vs. Ordinal Data:** Nominal data (Gender) has no math order and must be One-Hot Encoded. Ordinal data (Low, Med, High) has natural intensity and should be Label Encoded (0, 1, 2). 
* **The "Unknown" Paradox:** If you map Ordinal data (0, 1, 2) but have missing values, map "Unknown" to an extreme outlier like `-1`. Tree models will isolate it into a separate branch without ruining the 0-1-2 math order.
* **Selective Scaling:** Never apply `StandardScaler` blindly to an entire dataset. Exclude target variables, dummy columns, and ordinal variables. Only scale continuous numbers (Hours, Age, etc.).

### ⚙️ Big Data Bottlenecks
* **Time Complexity limits:** Algorithms like SVM and KNN have $O(n^2)$ time complexity. They will crash on datasets >100,000 rows.
* **The Kaggle Kings:** XGBoost, LightGBM, and CatBoost. These are Tree-based Gradient Boosting frameworks engineered to chew through millions of rows using advanced multithreading.
* **CatBoost's Native Handling:** CatBoost's superpower is its ability to eat raw categorical text natively without One-Hot Encoding, using Ordered Target Statistics to avoid cheating. It also automatically routes numerical `NaNs` through its trees, requiring zero imputation.

### 🏃‍♂️ Training Workflow
* **FAST_DEV_RUN (Dry Runs):** Never run a multi-hour training loop blindly. Create a switch that runs the model on 2 iterations to ensure the CSV saves correctly and there are no syntax errors at the end of the script.
* **Early Stopping:** Tell the model to monitor the validation score. If it stops improving for 100 trees, stop training automatically to prevent overfitting and save time.
* **Data Mutation (`.copy()`):** In Pandas, always use `.copy()` when passing a DataFrame into a feature engineering function to prevent permanently destroying the original raw data in RAM.

### 🏆 The Kaggle Meta (Grandmaster Strategies)
* **Feature Ablation & Residuals:** The best features are often hidden math. Subtract known hours from total hours to find "Residuals" (slack). Divide hours by total hours to find "Ratios" (percentages of daily life).
* **OOF (Out-Of-Fold) Predictions:** Saving the validation predictions from a Stratified K-Fold loop. This gives you an honest score on 100% of the training data without data leakage.
* **Blending / Ensembling:** You don't need to combine Python code to combine models. You can simply take the CSV outputs of two different models, average their probabilities together, and submit the blended CSV. 
* **Agentic Data Science (The AI Swarm):** Modern Grandmasters act as Orchestrators. They use scripts to deploy swarms of LLMs (GPT, Claude, DeepSeek) to autonomously brainstorm features, write code, and battle each other for the highest CV score.

NaN (Not a Number): The standard way programming languages (like Python/Pandas) represent missing or empty data.
Imputation: The process of replacing missing data with substituted values (like averages, medians, or predictions) so your AI doesn't crash when it reads the data.
Threshold (thresh): A limit you set in code. For dropping data, thresh=10 means "only keep rows that pass the test of having at least 10 valid data points."
KNN Imputation (K-Nearest Neighbors): A smart way to fill missing data. If User A is missing their gaming_hours, the AI finds 5 other users who have similar age, sleep hours, and screen time to User A, and copies their homework (averages their gaming hours) to fill User A's blank.
Categorical Encoding: Exactly what you just did! It’s the process of taking words (Male, Female, Other) and turning them into numbers (0.0, 1.0, 2.0) because Machine Learning models can only do math, and they can't do math on letters.
fit_transform: You will see this everywhere in Machine Learning.
Fit means "look at the data and learn the math/patterns."
Transform means "actually apply the changes to the data."
fit_transform does both at the exact same time!
NumPy Array vs. Pandas DataFrame: A Pandas DataFrame is like an Excel sheet (it has nice column names and row numbers). Scikit-learn tools (like KNN) hate column names, so they strip them away and return a NumPy Array, which is just a raw, naked grid of numbers. We always have to convert it back to a DataFrame so we can read it easily.
One-Hot Encoding (OHE): The process of taking a category (like Gender) and turning it into multiple Yes/No (1 or 0) columns. This stops the AI from accidentally thinking that "Female (2)" is mathematically greater than "Male (1)".
Big O Notation / Scalability: A concept in computer science that describes how much longer an algorithm takes as data gets bigger. KNN is notoriously slow for big data because it doesn't scale well.
pd.get_dummies(): The absolute best command in Pandas for One-Hot Encoding. It automatically takes a column, splits it into multiple columns based on the categories, and fills them with 1s and 0s.
IterativeImputer: An algorithmic imputer that treats every single column as its own mini Machine Learning target. It looks at Age, Screen Time, and Sleep to predict missing Gaming Hours. Then it looks at Age, Gaming Hours, and Sleep to predict missing Screen Time. It loops around ("iterates") until everything is filled perfectly.
Knowledge Entry #37: The "Hidden Categories" Problem. When trying to engineer a total sum from sub-categories (e.g., Total Screen Time = Gaming + Social Media), it usually fails in real-world data because there are always unrecorded activities (like watching videos or texting). You cannot force a perfect mathematical formula on human behavior.
Knowledge Entry #38: Small Data vs. Big Data Imputation. In small datasets (<10k rows), manual proxy imputation (like using Titles for Age) is great. In massive datasets (>500k rows), we rely on scalable methods like Grouped Medians, treating missing text as an "Unknown" category, or Algorithmic Imputation (MICE).
Knowledge Entry #39: The Ordinality Problem (Label vs. One-Hot Encoding). If you encode text like Gender or Colors into 0, 1, 2 (Label Encoding), math-based algorithms will assume 2 is greater or more important than 1. To prevent this, we use One-Hot Encoding (creating separate True/False columns for each category) so all categories are treated equally.
Knowledge Entry #40: The Grouped Imputation Trap. When filling missing numbers using a grouped average/median, you must never group by your Target Variable (the thing you are trying to predict). You won't have the target variable in the real-world test data, so your pipeline will crash.
Nominal Data: Categories with no order (e.g., Male/Female, Blue/Red, Apple/Orange). You must use One-Hot Encoding (creating new True/False columns).
Ordinal Data: Categories with a strict logical order (e.g., Low/Medium/High, Bad/Good/Excellent). You should use Label Encoding (0, 1, 2) because the math actually represents real-world intensity.
Knowledge Entry #61: OOF (Out-Of-Fold) Predictions. The predictions a model makes on the validation chunk of data that it was not allowed to see during training. Saving these predictions is crucial for Ensembling/Blending later.
Knowledge Entry #62: AUC (Area Under the ROC Curve). An evaluation metric that is often better than Accuracy for imbalanced datasets. It measures the model's ability to rank a positive case (Addicted) higher than a negative case (Not Addicted).
Knowledge Entry #63: Pipeline Templates (Config Variables). Professional code uses configuration variables at the very top (like TARGET = 'label') so the entire notebook can be instantly recycled for completely different machine learning projects without rewriting the core logic.
Knowledge Entry #64: Algorithmic Native Handling. You don't have to write code to handle missing numerical data if you are using advanced Gradient Boosters (CatBoost/LightGBM/XGBoost). Their internal C++ source code is already programmed to natively process NaN as a valid feature value.
Knowledge Entry #65: Data Mutation & .copy(). In Pandas, always use .copy() when passing a DataFrame into a function to engineer features. If you modify the DataFrame directly, you permanently alter the original raw data in memory (Mutation), which can lead to disastrous bugs if you need to re-run a cell.
🚀 The Final Code Block?
Knowledge Entry #66: FAST_DEV_RUN (Dry Runs). Before running a machine learning script that takes hours to complete, always build a "fast mode" switch that trains on tiny data or very few iterations. This allows you to debug code errors instantly without waiting hours.
Knowledge Entry #67: Early Stopping. A technique used in Gradient Boosting and Neural Networks. Instead of blindly training for a set number of rounds, you tell the model to monitor the validation score and stop automatically if the score stops improving. This prevents Overfitting and saves compute time.
Knowledge Entry #68: Stratified Splitting. When dealing with imbalanced datasets (e.g., 71% Positive, 29% Negative), normal random splitting can create biased chunks. Stratification guarantees that the original ratio is perfectly preserved across all train/test splits.
Knowledge Entry #69: Multithreading (thread_count=-1). Advanced ML algorithms allow you to specify how many CPU cores to use. Setting it to -1 tells Python to use 100% of your computer's available cores, maximizing speed but turning your laptop into a temporary space heater.
Knowledge Entry #70: The "Dry Run" Paradigm. Never run a massive, multi-hour script without doing a "Dry Run" (Fast Dev Run) first. Testing the entire pipeline (from data loading to CSV saving) on a tiny fraction of the data guarantees you won't lose hours of compute time to a silly syntax error at the very end.
Knowledge Entry #74: Late Submissions. On Kaggle, once a competition's deadline passes, the leaderboard is frozen in stone. However, you can still submit predictions to test your skills and see how you would have ranked. These are called Late Submissions and are entirely for educational purposes.
Knowledge Entry #75: Weighted Ensembling. When blending two CSVs together, you don't always have to do a 50/50 average. If Model A is much stronger than Model B, you can use a Weighted Average (e.g., (A * 0.7) + (B * 0.3)) to give the stronger model more voting power.