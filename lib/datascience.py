import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso

def chisq_ai_dropout():
    df = pd.read_csv('./AI Tools Adoption.csv')

    contingency = pd.crosstab(df['ai_use_purpose'], df['dropout_flag'])
    chi2, p, dof, ex = chi2_contingency(contingency)
    print(f"p-value = {p}")

def regressionPredicttestScore():
    df = pd.read_csv('./AI Tools Adoption.csv')

    numeric_cols = [
        'ai_use_frequency_per_week', 'session_count',
        'subjectivity',
        'adaptive_difficulty_start', 'adaptive_difficulty_end',
        'perceived_usefulness', 'perceived_ease_of_use',
        'trust_score', 'adoption_intention', 'pre_test_score',
        'ethical_concern_privacy', 'ethical_concern_plagiarism'
    ]

    X = df[numeric_cols]  # predictors

    y = df['post_test_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R²:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("Coefficients:", model.coef_)

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Post-Test Score")
    plt.ylabel("Predicted Post-Test Score")
    plt.title("Predicted vs Actual Scores")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal line
    plt.show()

    # Lasso regression (alpha controls regularization strength)
    lasso = Lasso(alpha=0.1)  # adjust alpha if too many/few features remain
    lasso.fit(X_train, y_train)

    # Predict
    y_pred = lasso.predict(X_test)

    # Evaluation
    print("R²:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Show selected features and coefficients
    selected_features = pd.Series(lasso.coef_, index=numeric_cols)
    selected_features = selected_features[selected_features != 0].sort_values(key=abs, ascending=False)
    print("\nSelected features and coefficients:\n", selected_features)
    results = pd.DataFrame({
        'Y_actual': y_test,
        'Y_predicted': y_pred
    })

    # Optional: show only first 10 rows to keep terminal clean
    print(results.head(10))