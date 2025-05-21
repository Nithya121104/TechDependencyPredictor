import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

# 1. Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)

    expected_cols = [
        'age', 'occupation', 'where_do_you_use_technology',
        'what_kind_of_technology_do_you_use_most',
        'What is your biggest concern about AI',
        'how_dependent_do_you_feel_on_technology_(scale_0_to_1)',
        'how_much_time_do_you_spend_using_technology_each_day_(in_hours)'
    ]

    for col in expected_cols:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return None, None, None

    df = df[expected_cols].dropna()

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df[['how_much_time_do_you_spend_using_technology_each_day_(in_hours)',
        'how_dependent_do_you_feel_on_technology_(scale_0_to_1)']] = scaler.fit_transform(
        df[['how_much_time_do_you_spend_using_technology_each_day_(in_hours)',
            'how_dependent_do_you_feel_on_technology_(scale_0_to_1)']]
    )

    # Rename columns for consistency
    df.rename(columns={
        'where_do_you_use_technology': 'where_use',
        'what_kind_of_technology_do_you_use_most': 'what_use',
        'how_much_time_do_you_spend_using_technology_each_day_(in_hours)': 'usage_amount',
        'how_dependent_do_you_feel_on_technology_(scale_0_to_1)': 'dependency_level',
        'What is your biggest concern about AI': 'how_use'  # approximate mapping
    }, inplace=True)

    return df, label_encoders, scaler

# 2. Train Model
def train_model(df):
    X = df[['age', 'occupation', 'how_use', 'where_use', 'what_use']]
    y = df[['dependency_level', 'usage_amount']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2_dep = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
    r2_usage = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

    print("Model trained with:")
    print(f" - R² score for dependency_level: {r2_dep:.4f}")
    print(f" - R² score for usage_amount: {r2_usage:.4f}")
    return model

# 3. Predict Future Dependency
def project_future_dependency(model, df):
    features = ['age', 'occupation', 'how_use', 'where_use', 'what_use']
    predictions = model.predict(df[features])
    df['predicted_dependency_now'] = predictions[:, 0]  # Only dependency_level

    df['predicted_dependency_50yrs'] = df['predicted_dependency_now'] + 0.3
    df['predicted_dependency_50yrs'] = df['predicted_dependency_50yrs'].clip(0, 1)
    return df

# 4. Visualization
def plot_dependency(df):
    current_avg = df['predicted_dependency_now'].mean()
    future_avg = df['predicted_dependency_50yrs'].mean()

    # Data for the bar graph
    labels = ['Current Dependency', 'Future Dependency (50 yrs)']
    values = [current_avg, future_avg]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['skyblue', 'salmon'])

    # Labels and title
    plt.ylabel("Average Technology Dependency Level")
    plt.title("Current vs Future Technology Dependency")

    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01 * max(values), f"{v:.2f}", ha='center', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
# 5. Generate Insights
def generate_insights(df):
    avg_now = df['predicted_dependency_now'].mean()
    avg_future = df['predicted_dependency_50yrs'].mean()
    print("\n--- Insights Report ---")
    print(f"Average current dependency: {avg_now:.2f}")
    print(f"Projected average dependency in 50 years: {avg_future:.2f}")
    if avg_future > 0.8:
        print("Warning: Technology over-dependency may impact cognitive abilities.")
    else:
        print("Dependency is growing but still manageable.")

# 6. Main
def main():
    filepath = 'influence_of_genai.csv'
    df, label_encoders, scaler = load_data(filepath)
    if df is None:
        return
    model = train_model(df)
    df = project_future_dependency(model, df)
    generate_insights(df)
    plot_dependency(df)


if __name__ == "__main__":
    main()

