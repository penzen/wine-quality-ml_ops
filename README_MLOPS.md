# 🍷 Wine Quality Prediction (MLOps Project)

## 📌 Problem Statement

This project aims to predict the **quality of white wine** based on its
chemical properties using a Machine Learning model.

Wine quality is influenced by multiple factors such as: - acidity -
sugar content - pH level - alcohol content

The goal is to learn the relationship:

    Wine Features → Wine Quality Score

This is a **regression problem**.

------------------------------------------------------------------------

## 🧰 Tools & Technologies

This project demonstrates an end-to-end ML workflow using:

-   **Keras** → Neural Network model
-   **NumPy / Pandas** → Data processing
-   **Scikit-learn** → Data splitting
-   **Hyperopt** → Hyperparameter tuning
-   **MLflow** → Experiment tracking and model logging

------------------------------------------------------------------------

## 🧠 Model Approach

### Data Pipeline

    Load Data → Split → Train → Validate → Tune → Log → Select Best Model

### Model Architecture

``` python
model = keras.Sequential([
    keras.Input([train_x.shape[1]]),
    keras.layers.Normalization(mean=mean, variance=var),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])
```

### Training Setup

``` python
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=params["lr"],
        momentum=params["momentum"]
    ),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()]
)
```

------------------------------------------------------------------------

## 🔍 Hyperparameter Tuning

Hyperopt is used to search for the best:

-   Learning Rate (`lr`)
-   Momentum (`momentum`)

Search space:

``` python
space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0)
}
```

10 different configurations are tested using:

``` python
fmin(...)
```

------------------------------------------------------------------------

## 📊 Results

Best performing model:

    Learning Rate: 0.0765
    Momentum:      0.25
    Validation RMSE: 0.715

### Interpretation

    Model error ≈ 0.71 wine quality points

This means predictions are on average within ±0.7 of the true value.

------------------------------------------------------------------------

## 📈 Experiment Tracking (MLflow)

MLflow is used to log:

-   Hyperparameters
-   RMSE metric
-   Trained models

Each Hyperopt trial is tracked as a nested run.

To launch UI:

``` bash
mlflow ui
```

Then open:

    http://127.0.0.1:5000

------------------------------------------------------------------------

## ▶️ How to Run the Project

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/wine-quality-mlops.git
cd wine-quality-mlops
```

------------------------------------------------------------------------

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 3. Run the script

``` bash
python Ml3.py
```

------------------------------------------------------------------------

### 4. Start MLflow UI

``` bash
mlflow ui
```

------------------------------------------------------------------------

## 📁 Project Structure

    project/
    │
    ├── Ml3.py
    ├── winequality-white.csv
    ├── README.md
    ├── requirements.txt

------------------------------------------------------------------------

## ⚠️ Notes

-   Model currently trains for only 3 epochs (for testing)
-   Increase epochs for better performance
-   Test dataset is not used yet for final evaluation

------------------------------------------------------------------------

## 🚀 Key Takeaways

This project demonstrates:

-   Neural network regression
-   Hyperparameter tuning with Hyperopt
-   Experiment tracking with MLflow
-   Clean MLOps workflow

------------------------------------------------------------------------

## 📌 Future Improvements

-   Increase training epochs
-   Add test set evaluation
-   Deploy model as API
-   Add feature scaling comparison (StandardScaler vs Keras
    Normalization)
