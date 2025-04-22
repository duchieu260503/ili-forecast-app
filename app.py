import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Load dataset
df = pd.read_csv("cleaned.csv").drop(columns=[col for col in pd.read_csv("cleaned.csv", nrows=1).columns if col.lower().startswith('unnamed')])
df["date"] = pd.to_datetime(df["date"])

# Display the dataset
st.title("ILI Forecast Comparison App")

with st.expander("ℹ️ What do these models do?"):
    st.markdown("""
    ### 🔍 Model Overview
    Understanding how each model works can help you choose the best one for your needs:

    - **Random Forest**
        - ✅ Strengths: Handles non-linear relationships well, robust to overfitting, works with small or large datasets.
        - ⚠️ Weaknesses: Slower to train, less interpretable.

    - **Linear Regression**
        - ✅ Strengths: Simple, fast, and easy to interpret. Good for linear trends.
        - ⚠️ Weaknesses: Not suitable for complex or non-linear data patterns.

    - **XGBoost**
        - ✅ Strengths: High accuracy, handles missing data, supports regularization, great for competitions.
        - ⚠️ Weaknesses: More complex and slower to train, especially with large parameter sets.

    Use this info to decide which model might suit your data best!
    """)

st.subheader("📋 Dataset Overview")
st.dataframe(df)
st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# Plot 2: Yearly average % Weighted ILI
import seaborn as sns
st.subheader("📅 Yearly Average % Weighted ILI")
df["year"] = df["date"].dt.year
yearly_avg = df.groupby("year")["weighted_ili"].mean().reset_index()
plt.figure(figsize=(12, 5))
sns.barplot(data=yearly_avg, x="year", y="weighted_ili", hue="year", palette="viridis", legend=False)
plt.xticks(rotation=90)
plt.title("Average % Weighted ILI by Year")
plt.xlabel("Year")
plt.ylabel("Avg. % Weighted ILI")
plt.tight_layout()
st.pyplot(plt)

# Create lag features
for lag in [1, 2, 3, 4, 5]:
    df[f"weighted_ili_lag_{lag}"] = df["weighted_ili"].shift(lag)

df.dropna(inplace=True)

# Define features and target
features = [f"weighted_ili_lag_{lag}" for lag in [1, 2, 3, 4, 5]]
X = df[features]
y = df["weighted_ili"]

# Train/test split
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Train models
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=10, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Compute metrics
rf_rmse = mean_squared_error(y_test, rf_preds)
lr_rmse = mean_squared_error(y_test, lr_preds)
xgb_rmse = mean_squared_error(y_test, xgb_preds)

rf_r2 = r2_score(y_test, rf_preds)
lr_r2 = r2_score(y_test, lr_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

# # Performance Summary
# st.subheader("📊 Model Performance Summary")
# st.markdown(f"- **Random Forest**: RMSE = {rf_rmse:.3f}, R² = {rf_r2:.3f}")
# st.markdown(f"- **Linear Regression**: RMSE = {lr_rmse:.3f}, R² = {lr_r2:.3f}")
# st.markdown(f"- **XGBoost**: RMSE = {xgb_rmse:.3f}, R² = {xgb_r2:.3f}")

# Plot: Predicted vs Actual

st.subheader("📈 Model Comparison: Predicted vs Actual")
plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(rf_preds, label="Random Forest", linestyle='--')
plt.plot(lr_preds, label="Linear Regression", linestyle='--')
plt.plot(xgb_preds, label="XGBoost", linestyle='--')
plt.xlabel("Weeks")
plt.ylabel("% Weighted ILI")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# # 📊 Seasonality Breakdown by Age Group
# st.subheader("🗓️ Weekly ILI Counts by Age Group (Heatmap)")
# df["year"] = df["date"].dt.year
# df["week"] = df["date"].dt.isocalendar().week

# age_cols = ["age_0_4", "age_5_24", "age_25_49", "age_50_64", "age_65"]
# for age_col in age_cols:
#     pivot = df.pivot_table(values=age_col, index="week", columns="year", aggfunc="sum")
#     plt.figure(figsize=(14, 6))
#     sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.1, linecolor="gray")
#     plt.title(f"Weekly ILI Cases for {age_col.replace('_', ' ').title()}")
#     plt.xlabel("Year")
#     plt.ylabel("Week")
#     st.pyplot(plt)


# # Residual Analysis
# st.subheader("🔍 Residual Analysis")
# residuals_df = pd.DataFrame({
#     "Random Forest": y_test.values - rf_preds,
#     "Linear Regression": y_test.values - lr_preds,
#     "XGBoost": y_test.values - xgb_preds
# })
# plt.figure(figsize=(14, 4))
# plt.boxplot([residuals_df[col] for col in residuals_df.columns], labels=residuals_df.columns)
# plt.title("Residuals Distribution")
# plt.ylabel("Residual Error")
# plt.grid(True)
# st.pyplot(plt)

# Forecasting section (kept as-is)

# 📊 Detailed Model Comparisons
st.subheader("🔍 Individual Model Evaluations")

def plot_model_vs_actual(model_name, y_true, y_pred):
    st.markdown(f"### {model_name}")
    col1, col2 = st.columns(2)
    
    with col1:
        rmse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        st.metric("RMSE", f"{rmse:.3f}")
        st.metric("R² Score", f"{r2:.3f}")

    with col2:
        plt.figure(figsize=(6, 3))
        plt.plot(y_true.values, label="Actual", linewidth=2)
        plt.plot(y_pred, label=model_name, linestyle='--')
        plt.title(f"{model_name}: Actual vs Predicted")
        plt.xlabel("Weeks")
        plt.ylabel("% Weighted ILI")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

plot_model_vs_actual("Random Forest", y_test, rf_preds)
plot_model_vs_actual("Linear Regression", y_test, lr_preds)
plot_model_vs_actual("XGBoost", y_test, xgb_preds)

# 🏁 Overall Summary
st.subheader("✅ Overall Model Performance Summary")
summary_df = pd.DataFrame({
    "Model": ["Random Forest", "Linear Regression", "XGBoost"],
    "RMSE": [mean_squared_error(y_test, rf_preds),
             mean_squared_error(y_test, lr_preds),
             mean_squared_error(y_test, xgb_preds)],
    "R²": [r2_score(y_test, rf_preds),
           r2_score(y_test, lr_preds),
           r2_score(y_test, xgb_preds)]
})

st.dataframe(summary_df.style.format({"RMSE": "{:.3f}", "R²": "{:.3f}"}))

st.subheader("🔮 Forecasting Next Weeks")
model_choice = st.selectbox("Choose model:", ["Random Forest", "Linear Regression", "XGBoost"])
forecast_weeks = st.slider("How many weeks to forecast?", 1, 100, 4)

# Get latest weighted_ili values
history = df["weighted_ili"].iloc[-5:].tolist()
predictions = []

# Recursive prediction
def predict_next_week(model, history):
    X_input = pd.DataFrame([{
        "weighted_ili_lag_1": history[-1],
        "weighted_ili_lag_2": history[-2],
        "weighted_ili_lag_3": history[-3],
        "weighted_ili_lag_4": history[-4],
        "weighted_ili_lag_5": history[-5],
    }])
    return model.predict(X_input)[0]

model_map = {
    "Random Forest": rf_model,
    "Linear Regression": lr_model,
    "XGBoost": xgb_model
}

model = model_map[model_choice]

for _ in range(forecast_weeks):
    pred = predict_next_week(model, history)
    predictions.append(pred)
    history.append(pred)

# Estimate ILI cases from %
latest_total_patients = df["total_patients"].iloc[-1]
forecast_dates = pd.date_range(start=df["date"].max() + pd.Timedelta(weeks=1), periods=forecast_weeks, freq="W")
forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "predicted_ili": predictions,
    "estimated_cases": [(p / 100) * latest_total_patients for p in predictions]
})

st.subheader("📈 Forecast Result")
st.dataframe(forecast_df.style.format({"predicted_ili": "{:.2f}", "estimated_cases": "{:.0f}"}))

# Download button
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Forecast CSV", data=csv, file_name="ili_forecast.csv", mime="text/csv")

# Compare forecast to recent history
st.subheader("📉 Historical vs Forecasted ILI Trend")
recent_history = df[["date", "weighted_ili"]].iloc[-12:].copy()
recent_history = recent_history.rename(columns={"weighted_ili": "predicted_ili"})
recent_history["estimated_cases"] = (recent_history["predicted_ili"] / 100) * latest_total_patients
combined_df = pd.concat([recent_history, forecast_df])

plt.figure(figsize=(12, 5))
plt.plot(combined_df["date"], combined_df["predicted_ili"], marker='o')
plt.axvline(x=forecast_df["date"].min(), color='gray', linestyle='--', label="Forecast Starts")
plt.title("% Weighted ILI: Historical and Forecast")
plt.xlabel("Date")
plt.ylabel("% Weighted ILI")
plt.legend()
st.pyplot(plt)