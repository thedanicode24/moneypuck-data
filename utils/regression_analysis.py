import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def print_summary(model, predictors):
    print(model.summary())
    for predictor in predictors:
        print(f"P-value ({predictor}): {model.pvalues[predictor]}")

def plot_linear_regression(df, predictor, target, 
                           figsize=(12,8), 
                           xlabel="Feature 1", 
                           ylabel="Feature 2", 
                           n_points=100):
    
    formula = target + " ~ " + predictor
    model = smf.ols(formula, data=df).fit()
    print_summary(model, [predictor])

    fit_xs = np.linspace(df[predictor].min(), df[predictor].max(), n_points)
    fit_df = pd.DataFrame({predictor: fit_xs})
    fit_ys = model.predict(fit_df)

    plt.figure(figsize=figsize)
    plt.scatter(df[predictor], df[target], marker=".", alpha=0.7, label="Observed data")
    plt.title("Linear Regression")
    plt.plot(fit_xs, fit_ys, color="C1", label="Best fit line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()

def fit_line(df, predictor, target, fit_xs):
    formula = f"{target} ~ {predictor}"
    model = smf.ols(formula, data=df)
    result = model.fit()
    fit_df = pd.DataFrame({predictor: fit_xs})
    fit_ys = result.predict(fit_df)
    return fit_ys

def resample(df):
    return df.sample(len(df), replace=True)

def plot_bootstrap_regression(df, predictor, target, n_bootstrap=1000, figsize=(12,8), xlabel="Predictor", ylabel="Target"):
    fit_xs = np.linspace(df[predictor].min(), df[predictor].max(), 200)
    fitted_ys = np.array([fit_line(resample(df), predictor, target, fit_xs) for _ in range(n_bootstrap)])
    
    low, median, high = np.percentile(fitted_ys, [5, 50, 95], axis=0)

    plt.figure(figsize=figsize)
    plt.title("Bootstrap Regression with 90% Confidence Interval")
    plt.scatter(df[predictor], df[target], marker=".", alpha=0.5, label="Observed data")
    plt.fill_between(fit_xs, low, high, color="C1", lw=0, alpha=0.2)
    plt.plot(fit_xs, median, color="C1", label="Bootstrap median fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def multiple_regression(df, target, predictors, summary=True):
    formula = f"{target} ~ {' + '.join(predictors)}"
    model = smf.ols(formula, data=df).fit()
    if summary:
        print_summary(model, predictors)