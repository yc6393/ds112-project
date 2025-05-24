"""PODS Project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


# sns.set_style("darkgrid")
# sns.set_context("paper")
# sns.set_palette("muted")

plt.style.use("seaborn-v0_8")
sns.set_context("paper", font_scale=2.0)
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

MIN_RATINGS = 3
ONLINE_THRESHOLD = 0.1  # 10% threshold


def preprocess_data():
    """Load and preprocess RateMyProfessor dataset."""
    # Assign column names
    num_df_columns = [
        "avg_rating",
        "avg_difficulty",
        "num_ratings",
        "pepper",
        "would_take_again",
        "online_ratings",
        "male",
        "female",
    ]
    qual_df_columns = ["major", "university", "state"]

    # Load datasets
    num_df = pd.read_csv("rmpCapstoneNum.csv", names=num_df_columns)
    qual_df = pd.read_csv("rmpCapstoneQual.csv", names=qual_df_columns)

    # Join datasets and pre-process
    df = pd.concat([num_df, qual_df], axis=1)
    df = (
        df.dropna(subset=["avg_rating"])
        .query("num_ratings >= @MIN_RATINGS")
        .assign(
            online_ratio=lambda x: x["online_ratings"] / x["num_ratings"],
            rating_per_difficulty=lambda x: x["avg_rating"]
            / x["avg_difficulty"],
            is_experienced=lambda x: x["num_ratings"]
            > x["num_ratings"].median(),
            log_experience=lambda x: np.log(x["num_ratings"]),
            sex=lambda x: np.select(
                [x["male"].eq(1), x["female"].eq(1)],
                ["Male", "Female"],
                default="Unknown",
            ),
            hot=lambda x: np.select(
                [x["pepper"].eq(1), x["pepper"].eq(0)],
                [True, False],
                default="Unknown",
            ),
        )
    )
    return df


def analyze_gender_bias(df):
    """Analyze gender bias in professor ratings."""
    # Overall gender comparison
    male_ratings = df[df["male"] == 1]["avg_rating"]
    female_ratings = df[df["female"] == 1]["avg_rating"]

    # Calculate overall means and difference
    male_mean = male_ratings.mean()
    female_mean = female_ratings.mean()
    overall_diff = male_mean - female_mean

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(male_ratings, female_ratings)

    # Create distribution plot
    plt.figure()
    sns.boxplot(
        data=df.query('sex!="Unknown"'), hue="sex", y="avg_rating", gap=0.1
    )
    plt.title("Professor Ratings by Gender")
    plt.tight_layout()

    # Create strata
    df["experience_level"] = pd.qcut(
        df["num_ratings"], q=3, labels=["Low", "Medium", "High"]
    )
    df["difficulty_level"] = pd.qcut(
        df["avg_difficulty"], q=3, labels=["Easy", "Medium", "Hard"]
    )

    # Initialize matrices
    experience_levels = ["Low", "Medium", "High"]
    difficulty_levels = ["Easy", "Medium", "Hard"]

    p_values = np.zeros((3, 3))
    raw_differences = np.zeros((3, 3))
    male_means = np.zeros((3, 3))
    female_means = np.zeros((3, 3))

    # Analyze each stratum
    for i, exp in enumerate(experience_levels):
        for j, diff in enumerate(difficulty_levels):
            stratum = df[
                (df["experience_level"] == exp)
                & (df["difficulty_level"] == diff)
            ]

            male_ratings = stratum[stratum["male"] == 1]["avg_rating"]
            female_ratings = stratum[stratum["female"] == 1]["avg_rating"]

            if len(male_ratings) >= 30 and len(female_ratings) >= 30:
                t_stat_strat, p_val = stats.ttest_ind(
                    male_ratings, female_ratings
                )
                raw_diff = male_ratings.mean() - female_ratings.mean()

                p_values[i, j] = p_val
                raw_differences[i, j] = raw_diff
                male_means[i, j] = male_ratings.mean()
                female_means[i, j] = female_ratings.mean()

    # Create heatmap visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # 1. P-values heatmap
    sns.heatmap(
        p_values,
        annot=True,
        fmt=".4f",
        cmap="RdYlBu_r",
        xticklabels=difficulty_levels,
        yticklabels=experience_levels,
        ax=ax1,
        vmin=0,
        vmax=1,
    )
    ax1.set_title("p-values\n(darker = more significant)")
    ax1.set_xlabel("Course Difficulty")
    ax1.set_ylabel("Experience Level")

    # 2. Raw differences heatmap
    sns.heatmap(
        raw_differences,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        xticklabels=difficulty_levels,
        yticklabels=experience_levels,
        ax=ax2,
        center=0,
    )
    ax2.set_title("Differences in Ratings\n(male mean - female mean)")
    ax2.set_xlabel("Course Difficulty")
    ax2.set_ylabel("Experience Level")

    plt.tight_layout()

    return (
        t_stat,
        p_value,
        male_mean,
        female_mean,
        overall_diff,
        p_values,
        raw_differences,
    )


def analyze_experience_effect(df):
    """Analyzes the relationship between teaching experience and ratings."""
    # Use log transform for experience due to right-skewed distribution
    # df["log_experience"] = np.log(df["num_ratings"])

    # Fit regression model
    X = df["log_experience"].values.reshape(-1, 1)
    y = df["avg_rating"].values

    model = stats.linregress(X.flatten(), y)

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with regression line
    sns.regplot(
        data=df,
        x="log_experience",
        y="avg_rating",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
    )

    ax.set_title("Effect of Experience on Ratings")
    ax.set_xlabel("Log Number of Ratings (Experience Proxy)")
    ax.set_ylabel("Average Rating")

    plt.tight_layout()

    return model


def analyze_rating_difficulty(df):
    """Analyzes the relationship between course difficulty and ratings."""
    # Fit regression model
    model = stats.linregress(df["avg_difficulty"], df["avg_rating"])

    # Create visualization
    # fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with regression line and hexbin for density
    sns.jointplot(
        data=df, x="avg_difficulty", y="avg_rating", kind="hex", height=6
    )
    plt.suptitle(
        "Relationship between Rating and Difficulty", y=1.02, fontsize=10
    )
    plt.figure(figsize=(6, 6))

    plt.tight_layout()

    return model


def analyze_online_effect(df):
    """Analyze how online teaching affects professor ratings."""
    # Approach 1: Correlation analysis
    correlation, p_value_corr = stats.pearsonr(
        df["online_ratio"], df["avg_rating"]
    )

    # Approach 2: Online vs traditional (>10% threshold)
    online = df[df["online_ratio"] > ONLINE_THRESHOLD]["avg_rating"]
    traditional = df[df["online_ratio"] <= ONLINE_THRESHOLD]["avg_rating"]

    t_stat, p_value = stats.ttest_ind(online, traditional)

    online_mean = online.mean()
    trad_mean = traditional.mean()
    mean_diff = online_mean - trad_mean

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # 1. Scatter plot with regression line
    sns.regplot(
        data=df,
        x="online_ratio",
        y="avg_rating",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
        ax=ax1,
    )
    ax1.set_title(
        f"Continuous Relationship\nr = {correlation:.3f}, p = {p_value_corr:.4f}"
    )
    ax1.set_xlabel("Proportion of Online Classes")
    ax1.set_ylabel("Average Rating")

    # 2. Box plot comparing online vs traditional
    df["online_focused"] = df["online_ratio"] > ONLINE_THRESHOLD
    sns.boxplot(data=df, hue="online_focused", y="avg_rating", gap=0.1, ax=ax2)
    ax2.set_title("Ratings Distribution: Online vs Traditional")
    ax2.set_xlabel("Online-focused (>10% online classes)")
    ax2.set_ylabel("Average Rating")

    plt.tight_layout()

    # Additional context
    n_online = len(online)
    n_trad = len(traditional)

    return (
        correlation,
        p_value_corr,
        t_stat,
        p_value,
        online_mean,
        trad_mean,
        mean_diff,
        n_online,
        n_trad,
    )


def analyze_would_take_again(df):
    """Analyze relationship between ratings and would-take-again proportion."""
    # Remove rows where would_take_again is missing
    df_clean = df.dropna(subset=["would_take_again"])

    # Correlation analysis
    correlation, p_value = stats.pearsonr(
        df_clean["would_take_again"], df_clean["avg_rating"]
    )

    # Fit regression model
    model = stats.linregress(
        df_clean["would_take_again"], df_clean["avg_rating"]
    )

    plt.figure(figsize=(10, 6))
    # Scatter plot with regression line
    sns.regplot(
        data=df_clean,
        x="would_take_again",
        y="avg_rating",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
    )

    plt.title(
        f"Ratings vs Would-Take-Again Percentage\nr = {correlation:.3f}, p = {p_value:.2e}"
    )
    plt.xlabel("Percentage Who Would Take Again")
    plt.ylabel("Average Rating")

    # Add regression equation
    equation = f"y = {model.slope:.3f}x + {model.intercept:.3f}"
    r2 = f"R² = {model.rvalue**2:.3f}"
    plt.text(
        0.05,
        0.95,
        equation + "\n" + r2,
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )

    plt.tight_layout()

    return model, len(df_clean)


def analyze_hotness_effect(df):
    """Analyze relationship between professor 'hotness' and ratings."""
    # Get ratings for hot and not-hot professors
    hot_ratings = df[df["pepper"] == 1]["avg_rating"]
    not_ratings = df[df["pepper"] == 0]["avg_rating"]

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(hot_ratings, not_ratings)

    # Calculate means and difference
    hot_mean = hot_ratings.mean()
    not_mean = not_ratings.mean()
    mean_diff = hot_mean - not_mean

    # Sample sizes
    n_hot = len(hot_ratings)
    n_not = len(not_ratings)

    # Create visualization
    plt.figure(figsize=(10, 6))

    # Boxplot
    sns.boxplot(data=df, hue="hot", y="avg_rating", gap=0.1)
    plt.title("Professor Ratings by Hotness")
    plt.ylabel("Average Rating")

    plt.tight_layout()

    return t_stat, p_value, hot_mean, not_mean, mean_diff, n_hot, n_not


def build_difficulty_model(df):
    """Build a regression model predicting ratings from difficulty only."""
    # Prepare data
    X = df["avg_difficulty"].values.reshape(-1, 1)
    y = df["avg_rating"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Create visualization
    plt.figure(figsize=(10, 6))

    # Scatter plot with regression line
    sns.regplot(
        data=df,
        x="avg_difficulty",
        y="avg_rating",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
    )

    plt.title("Ratings vs Difficulty")
    plt.xlabel("Average Difficulty")
    plt.ylabel("Average Rating")

    # Add model details
    equation = f"y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}"
    metrics = f"Test R² = {test_r2:.3f}\nTest RMSE = {test_rmse:.3f}"
    plt.text(
        0.05,
        0.15,
        equation + "\n" + metrics,
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )

    plt.tight_layout()

    return model, (train_r2, test_r2, train_rmse, test_rmse)


def build_full_model(df, include_would_take_again=False):
    """Build a regression model predicting ratings from available factors."""
    # Create log_experience feature
    df["log_experience"] = np.log(df["num_ratings"])

    # Prepare features
    features = [
        "avg_difficulty",
        "log_experience",
        "pepper",
        "online_ratio",
        "male",
    ]

    if include_would_take_again:
        features.append("would_take_again")
        df = df.dropna(subset=["would_take_again"])

    # Check for collinearity
    correlation_matrix = df[features].corr()
    vif_data = pd.DataFrame()
    for feature in features:
        X_temp = df[features].drop(feature, axis=1)
        y_temp = df[feature]
        r2 = LinearRegression().fit(X_temp, y_temp).score(X_temp, y_temp)
        vif = 1 / (1 - r2)
        vif_data = pd.concat(
            [vif_data, pd.DataFrame({"Feature": [feature], "VIF": [vif]})]
        )

    # Prepare data
    X = df[features]
    y = df["avg_rating"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Create coefficient summary
    coef_df = pd.DataFrame(
        {
            "Feature": features,
            "Coefficient": model.coef_,
            "Abs_Coefficient": abs(model.coef_),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Actual vs Predicted plot
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5, ax=ax1)
    ax1.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        alpha=0.8,
    )
    ax1.set_xlabel("Actual Rating")
    ax1.set_ylabel("Predicted Rating")
    ax1.set_title("Actual vs Predicted Ratings")

    # 2. Coefficient plot
    sns.barplot(data=coef_df, x="Coefficient", y="Feature", ax=ax2)
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax2.set_title("Standardized Coefficients")

    # Add model metrics
    metrics_text = (
        f"Test R² = {test_r2:.3f}\n"
        f"Test RMSE = {test_rmse:.3f}\n"
        f"n = {len(df)}"
    )
    ax1.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=1),
    )

    plt.tight_layout()

    return model, (train_r2, test_r2, train_rmse, test_rmse), vif_data, coef_df


def build_pepper_classifier_simple(df):
    """Build a classifier predicting 'pepper' from average rating only."""
    # Prepare data
    X = df["avg_rating"].values.reshape(-1, 1)
    y = df["pepper"].astype(int).values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Check class balance
    class_counts = np.bincount(y)
    class_balance = class_counts / len(y)

    # Fit model
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_prob)
    confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    ax1.plot(
        fpr,
        tpr,
        color="blue",
        lw=2,
        label=f"ROC curve (AUC = {auc_score:.3f})",
    )
    ax1.plot([0, 1], [0, 1], color="red", linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    # 2. Distribution of ratings by pepper status
    sns.kdeplot(data=df, x="avg_rating", hue="pepper", ax=ax2)
    ax2.set_title("Rating Distribution by Pepper Status")
    ax2.set_xlabel("Average Rating")
    ax2.set_ylabel("Density")

    # Add metrics to plot
    metrics_text = (
        f"Accuracy: {accuracy:.3f}\n"
        f"Sensitivity: {sensitivity:.3f}\n"
        f"Specificity: {specificity:.3f}\n"
        f"AUC: {auc_score:.3f}"
    )
    ax1.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=1),
    )

    plt.tight_layout()

    return (
        model,
        (accuracy, sensitivity, specificity, auc_score),
        class_balance,
    )


def build_pepper_classifier_full(df):
    """Build a classifier predicting 'pepper' from all available factors."""
    # Prepare features
    features = [
        "avg_rating",
        "avg_difficulty",
        "log_experience",
        "online_ratio",
        "male",
    ]

    X = df[features]
    y = df["pepper"].astype(int).values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check class balance
    class_counts = np.bincount(y)
    class_balance = class_counts / len(y)

    # Fit model
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_prob)
    confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion.ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Create coefficient summary
    coef_df = pd.DataFrame(
        {
            "Feature": features,
            "Coefficient": model.coef_[0],
            "Abs_Coefficient": abs(model.coef_[0]),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    ax1.plot(
        fpr,
        tpr,
        color="blue",
        lw=2,
        label=f"ROC curve (AUC = {auc_score:.3f})",
    )
    ax1.plot([0, 1], [0, 1], color="red", linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    # 2. Feature importance plot
    sns.barplot(data=coef_df, x="Coefficient", y="Feature", ax=ax2)
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax2.set_title("Feature Coefficients (Standardized)")

    # Add metrics to plot
    metrics_text = (
        f"Accuracy: {accuracy:.3f}\n"
        f"Sensitivity: {sensitivity:.3f}\n"
        f"Specificity: {specificity:.3f}\n"
        f"AUC: {auc_score:.3f}"
    )
    ax1.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=1),
    )

    plt.tight_layout()

    return (
        model,
        (accuracy, sensitivity, specificity, auc_score),
        class_balance,
        coef_df,
    )


def analyze_major_effect(df):
    """Analyze the relationship between academic major and professor ratings."""
    # Get top 20 most common majors for meaningful analysis
    top_majors = df["major"].value_counts().head(20).index
    df_filtered = df[df["major"].isin(top_majors)]

    # Calculate summary statistics by major
    major_stats = (
        df_filtered.groupby("major")["avg_rating"]
        .agg(["mean", "std", "count"])
        .round(3)
        .sort_values("mean", ascending=True)
    )  # Sort ascending for bottom-to-top plot

    # Perform one-way ANOVA
    majors_list = [
        group["avg_rating"].values
        for name, group in df_filtered.groupby("major")
    ]
    f_stat, p_value = stats.f_oneway(*majors_list)

    # Create visualization
    plt.figure(figsize=(10, 10))

    # Horizontal boxplot
    sns.boxplot(
        data=df_filtered, y="major", x="avg_rating", order=major_stats.index
    )
    plt.title("Professor Ratings by Academic Major")
    plt.xlabel("Average Rating")
    plt.ylabel("Major")

    # Add ANOVA results
    plt.text(
        0.02,
        0.98,
        f"ANOVA: p = {p_value:.2e}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=1),
    )

    plt.tight_layout()

    return major_stats, (f_stat, p_value)


if __name__ == "__main__":

    # Set random seed
    np.random.seed(16378429)

    df = preprocess_data()

    # QUESTION 1 ##############################################################
    (
        t_stat,
        p_value,
        male_mean,
        female_mean,
        overall_diff,
        strat_p_values,
        strat_differences,
    ) = analyze_gender_bias(df)
    print("\n")
    print("==================================================================")
    print("QUESTION 1 - Gender Bias")
    print("==================================================================")
    print("\nOverall Gender Comparison:")
    print(f"Male mean rating: {male_mean:.3f}")
    print(f"Female mean rating: {female_mean:.3f}")
    print(f"Difference (Male - Female): {overall_diff:.3f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.2e}")

    # QUESTION 2 ##############################################################
    print("\n")
    print("==================================================================")
    print("QUESTION 2 - Rating vs Experience")
    print("==================================================================")
    model = analyze_experience_effect(df)
    print("\nExperience Effect Regression Analysis:")
    print(f"Slope: {model.slope:.3f}")
    print(f"Intercept: {model.intercept:.3f}")
    print(f"R-squared: {model.rvalue**2:.3f}")
    print(f"P-value: {model.pvalue:.2e}")

    # QUESTION 3 ##############################################################
    model = analyze_rating_difficulty(df)
    print("\n")
    print("==================================================================")
    print("QUESTION 3 - Rating vs Difficulty")
    print("==================================================================")
    print("\nRating Difficulty Regression Analysis:")
    print(f"Slope: {model.slope:.3f}")
    print(f"Intercept: {model.intercept:.3f}")
    print(f"R-squared: {model.rvalue**2:.3f}")
    print(f"P-value: {model.pvalue:.2e}")
    print(f"Correlation: {model.rvalue:.3f}")

    # QUESTION 4 ##############################################################
    (
        corr,
        p_val_corr,
        t_stat,
        p_val,
        online_mean,
        trad_mean,
        diff,
        n_online,
        n_trad,
    ) = analyze_online_effect(df)
    print("\n")
    print("==================================================================")
    print("QUESTION 4 - Rating vs Online Modality")
    print("==================================================================")
    print("\nApproach 1: Correlation Analysis")
    print(f"Correlation: {corr:.3f}")
    print(f"P-value: {p_val_corr:.2e}")
    print("\nApproach 2: Online vs Traditional Comparison (10% threshold)")
    print(f"Online-focused professors mean rating: {online_mean:.3f}")
    print(f"Traditional professors mean rating: {trad_mean:.3f}")
    print(f"Difference (Online - Traditional): {diff:.3f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.2e}")
    print("\nSample sizes:")
    print(f"Online-focused professors (>10% online): {n_online}")
    print(f"Traditional professors (≤10% online): {n_trad}")

    # QUESTION 5 ##############################################################
    model, n = analyze_would_take_again(df)
    print("\n")
    print("==================================================================")
    print("QUESTION 5 - Rating vs 'Would take again'")
    print("==================================================================")
    print(f"\nRegression Analysis Results (n={n}):")
    print(f"Slope: {model.slope:.3f}")
    print(f"Intercept: {model.intercept:.3f}")
    print(f"R-squared: {model.rvalue**2:.3f}")
    print(f"P-value: {model.pvalue:.2e}")
    print(f"Correlation: {model.rvalue:.3f}")

    # QUESTION 6 ##############################################################
    t_stat, p_val, hot_mean, not_mean, diff, n_hot, n_not = (
        analyze_hotness_effect(df)
    )
    print("\n")
    print("==================================================================")
    print("QUESTION 6 - Hotness Effect on Rating")
    print("==================================================================")
    print("\nHotness Effect Analysis:")
    print(f"'Hot' professors mean rating: {hot_mean:.3f}")
    print(f"Other professors mean rating: {not_mean:.3f}")
    print(f"Difference (Hot - Not): {diff:.3f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.2e}")
    print("\nSample sizes:")
    print(f"'Hot' professors: {n_hot}")
    print(f"Other professors: {n_not}")

    # QUESTION 7 ##############################################################
    model, metrics = build_difficulty_model(df)
    train_r2, test_r2, train_rmse, test_rmse = metrics
    print("\n")
    print("==================================================================")
    print("QUESTION 7 - Regression for Ratings vs Difficulty")
    print("==================================================================")
    print("\nRegression Model Results:")
    print(f"Coefficient (slope): {model.coef_[0]:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")
    print("\nTraining Metrics:")
    print(f"R-squared: {train_r2:.3f}")
    print(f"RMSE: {train_rmse:.3f}")
    print("\nTest Metrics:")
    print(f"R-squared: {test_r2:.3f}")
    print(f"RMSE: {test_rmse:.3f}")

    # QUESTION 8 ##############################################################
    # Model without would_take_again
    print("\n")
    print("==================================================================")
    print("QUESTION 8 - Regression for Ratings vs Everything")
    print("==================================================================")

    print("\nModel WITHOUT would_take_again:")
    model1, metrics1, vif1, coef1 = build_full_model(
        df, include_would_take_again=False
    )
    train_r2, test_r2, train_rmse, test_rmse = metrics1
    print("\nFeature Coefficients (standardized):")
    print(coef1.to_string(index=False))
    print("\nVariance Inflation Factors:")
    print(vif1.to_string(index=False))
    print(f"\nTest Metrics (n={len(df)}):")
    print(f"R-squared: {test_r2:.3f}")
    print(f"RMSE: {test_rmse:.3f}")

    # Model with would_take_again
    print("\n\nModel WITH would_take_again:")
    model2, metrics2, vif2, coef2 = build_full_model(
        df, include_would_take_again=True
    )
    train_r2, test_r2, train_rmse, test_rmse = metrics2
    print("\nFeature Coefficients (standardized):")
    print(coef2.to_string(index=False))
    print("\nVariance Inflation Factors:")
    print(vif2.to_string(index=False))
    print(f"\nTest Metrics (n={len(df.dropna(subset=['would_take_again']))}):")
    print(f"R-squared: {test_r2:.3f}")
    print(f"RMSE: {test_rmse:.3f}")

    # QUESTION 9 ##############################################################
    model, metrics, class_balance = build_pepper_classifier_simple(df)
    accuracy, sensitivity, specificity, auc_score = metrics
    print("\n")
    print("==================================================================")
    print("QUESTION 9 - Pepper vs Ratings Classificaiton")
    print("==================================================================")
    print("\nClassification Model Results:")
    print("\nClass Balance:")
    print(f"No Pepper: {class_balance[0]:.1%}")
    print(f"Has Pepper: {class_balance[1]:.1%}")
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.3f}")
    print(f"Specificity (True Negative Rate): {specificity:.3f}")
    print(f"AUC-ROC: {auc_score:.3f}")
    print("\nModel Parameters:")
    print(f"Intercept: {model.intercept_[0]:.3f}")
    print(f"Coefficient: {model.coef_[0][0]:.3f}")

    # QUESTION 10 #############################################################
    model, metrics, class_balance, coef_df = build_pepper_classifier_full(df)
    accuracy, sensitivity, specificity, auc_score = metrics
    print("\n")
    print("==================================================================")
    print("QUESTION 10 - General Pepper Classification")
    print("==================================================================")
    print("\nClassification Model Results:")
    print("\nClass Balance:")
    print(f"No Pepper: {class_balance[0]:.1%}")
    print(f"Has Pepper: {class_balance[1]:.1%}")
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.3f}")
    print(f"Specificity (True Negative Rate): {specificity:.3f}")
    print(f"AUC-ROC: {auc_score:.3f}")

    print("\nFeature Coefficients (standardized):")
    print(coef_df.to_string(index=False))

    # QUESTION EXTRA ##########################################################
    major_stats, (f_stat, p_value) = analyze_major_effect(df)
    print("\n")
    print("==================================================================")
    print("EXTRA CREDIT QUESTION")
    print("==================================================================")
    print("\nRatings by Major:")
    print(major_stats)
    print(f"\nANOVA Results:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_value:.2e}")
