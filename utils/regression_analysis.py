from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

def predict(result, xs):
    ys = result.intercept + result.slope * xs
    return ys

def plot_linear_regression(df1, df2, figsize=(12,8), xlabel="Feature 1", ylabel="Feature 2"):
    result = linregress(df1, df2)
    fit_xs = np.linspace(np.min(df1), np.max(df1))
    fit_ys = predict(result, fit_xs)

    print_metrics(df1, df2, result)

    plt.figure(figsize=figsize)
    plt.scatter(df1, df2, marker=".", alpha=0.7, label="Observed data")
    plt.title("Linear Regression")
    plt.plot(fit_xs, fit_ys, color="C1", label="Best fit line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()

def compute_residuals(result, xs, y):
    y_pred = predict(result, xs)
    return y - y_pred

def compute_mse(result, xs, y):
    residuals = compute_residuals(result, xs, y)
    return np.mean(residuals**2)

def print_metrics(df1, df2, result):
    print(f"Mean Squared Error: {compute_mse(result, df1, df2):.3f}")
    print(f"Coefficient of determination: {result.rvalue**2:.3f}")
    print(f"Standard error: {result.stderr:.3f}")


def fit_line(df, feat1, feat2, fit_xs):
    xs, ys = df[feat1], df[feat2]
    result = linregress(xs, ys)
    fit_ys = predict(result, fit_xs)
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