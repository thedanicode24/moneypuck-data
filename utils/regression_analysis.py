import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.thinkstats import display_summary

def plot_linear_regression(df, feat1, feat2, figsize=(12,8), xlabel="Feature 1", ylabel="Feature 2", n_points=100):
    formula = feat2 + " ~ " + feat1
    model = smf.ols(formula, data=df)
    result = model.fit()
    fit_xs = np.linspace(df[feat1].min(), df[feat1].max(), n_points)
    fit_df = pd.DataFrame({feat1: fit_xs})
    fit_ys = result.predict(fit_df)

    display_summary(result)
    print(f"P-value: {result.pvalues[feat1]}")

    plt.figure(figsize=figsize)
    plt.scatter(df[feat1], df[feat2], marker=".", alpha=0.7, label="Observed data")
    plt.title("Linear Regression")
    plt.plot(fit_xs, fit_ys, color="C1", label="Best fit line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()

def fit_line(df, feat1, feat2, fit_xs):
    formula = f"{feat2} ~ {feat1}"
    model = smf.ols(formula, data=df)
    result = model.fit()
    fit_df = pd.DataFrame({feat1: fit_xs})
    fit_ys = result.predict(fit_df)
    return fit_ys

def resample(df):
    return df.sample(len(df), replace=True)

def plot_bootstrap_regression(df, feature1, feature2, n_bootstrap=1000, figsize=(12,8), xlabel="Feature 1", ylabel="Feature 2"):
    fit_xs = np.linspace(df[feature1].min(), df[feature1].max(), 200)
    fitted_ys = np.array([fit_line(resample(df), feature1, feature2, fit_xs) for _ in range(n_bootstrap)])
    
    low, median, high = np.percentile(fitted_ys, [5, 50, 95], axis=0)

    plt.figure(figsize=figsize)
    plt.title("Bootstrap Regression with 90% Confidence Interval")
    plt.scatter(df[feature1], df[feature2], marker=".", alpha=0.5, label="Observed data")
    plt.fill_between(fit_xs, low, high, color="C1", lw=0, alpha=0.2)
    plt.plot(fit_xs, median, color="C1", label="Bootstrap median fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()