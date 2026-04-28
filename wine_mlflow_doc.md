# Wine Quality Prediction (Keras + Hyperopt + MLflow)

## Overview
This project builds a regression model using a neural network to predict wine quality.  
It also demonstrates **hyperparameter tuning (Hyperopt)** and **experiment tracking (MLflow)**.

Keras → builds the neural network
NumPy → numerical operations
Pandas → loads dataset
Hyperopt → hyperparameter tuning
Scikit-learn → data splitting
MLflow → experiment tracking

------------------------------------------------------------------------

## Data Loading

``` python
data = pd.read_csv("winequality-white.csv", sep=";")
```

Loads dataset with correct separator.

------------------------------------------------------------------------

## Train-Test Split

``` python
train, test = train_test_split(data, test_size=0.25, random_state=42)
```

-   75% training
-   25% test

------------------------------------------------------------------------

## Feature / Target Split

``` python
train_x = train.iloc[:, :-1].values
train_y = train.iloc[:, -1].values
```

-   Features = all columns except last
-   Target = last column (quality)

------------------------------------------------------------------------

## Validation Split

``` python
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.20, random_state=42
)
```

Final: - Train: 60% - Validation: 15% - Test: 25%

------------------------------------------------------------------------

## Model Function

``` python
def train_model(params, epocs, train_x, train_y, valid_x, valid_y, test_x, test_y):
```

Trains one model per hyperopt trial.

------------------------------------------------------------------------

## Normalization

``` python
mean = np.mean(train_x, axis=0)
var = np.var(train_x, axis=0)
```

Used to scale input features:

    (x - mean) / sqrt(var)

------------------------------------------------------------------------

## Model Architecture

``` python
model = keras.Sequential([
    keras.Input([train_x.shape[1]]),
    keras.layers.Normalization(mean=mean, variance=var),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])
```

Structure: Input → Normalize → Dense(64) → Output

------------------------------------------------------------------------

## Compile Model

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

-   Loss: MSE
-   Metric: RMSE

------------------------------------------------------------------------

## Training

``` python
model.fit(train_x, train_y,
          validation_data=(valid_x, valid_y),
          epochs=epocs,
          batch_size=64)
```

------------------------------------------------------------------------

## Evaluation

``` python
eval_result = model.evaluate(valid_x, valid_y)
eval_rmse = eval_result[1]
```

-   eval_result = \[loss, rmse\]
-   RMSE used for optimization

------------------------------------------------------------------------

## MLflow Logging

``` python
mlflow.log_params(params)
mlflow.log_metric("eval_rmse", eval_rmse)
```

------------------------------------------------------------------------

## Save Model

The signature tells MLflow what the model input and output look like.
It records:

input shape
input type
output shape
output type

This is useful when saving and deploying the model.

``` python
mlflow.tensorflow.log_model(model, "model", signature=signature)
```

------------------------------------------------------------------------

## Hyperopt Objective

``` python
def objective(params):
    return train_model(...)
```

------------------------------------------------------------------------

## Search Space

``` python
space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0)
}
```

------------------------------------------------------------------------

## Run Optimization

``` python
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)
```

------------------------------------------------------------------------

## Best Model

``` python
best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
```

------------------------------------------------------------------------

## Final Logging

``` python
mlflow.log_params(best)
mlflow.log_metric("eval_rmse", best_run["loss"])
```

------------------------------------------------------------------------

## Result

Example:

    best parameter {'lr': 0.0765, 'momentum': 0.25}
    best eval rmse 0.715

------------------------------------------------------------------------

## Key Idea

Pipeline:

    Data → Train Model → Evaluate → Tune (Hyperopt) → Track (MLflow)
