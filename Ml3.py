import keras 
import numpy as np 
import pandas as pd 
from hyperopt import STATUS_OK,Trials,fmin,hp,tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # performs normalization 
import mlflow
from mlflow.models import infer_signature


data = pd.read_csv("winequality-white.csv",sep=";")


# 1) First split: full data -> train + test
train,test = train_test_split(data, test_size= 0.25, random_state= 42)



# 2) Separate features and target for training data
train_x = train.iloc[:,:-1].values
train_y = train.iloc[:,-1].values


# 3) Separate features and target for test data

test_x = test.iloc[:,:-1].values
test_y = test.iloc[:,-1].values


# 4) Second split: train -> train + validation
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x,
    train_y,
    test_size=0.20,
    random_state=42
)


signature = infer_signature(train_x,train_y)


#ANN model 

def train_model(params,epocs,train_x,train_y,valid_x,valid_y,test_x,test_y):

    # define the model aricitecture 
    mean = np.mean(train_x, axis = 0 )
    var = np.var(train_x,axis = 0)

    model = keras.Sequential(
    [
        keras.Input([train_x.shape[1]]), # 11 input fetures, this is the input layer
        keras.layers.Normalization(mean = mean, variance = var),
        keras.layers.Dense(64,activation= 'relu'), # number of hidden neaurons are 64 
        keras.layers.Dense(1) # this is the output layer
    ])

    model.compile(optimizer= keras.optimizers.SGD(learning_rate= params["lr"],momentum= params["momentum"]), 
        loss = keras.losses.MeanSquaredError(),
        metrics= [keras.metrics.RootMeanSquaredError()]) # this is returned in the eval_rmse 
    
    with mlflow.start_run(nested= True):
        model.fit(train_x,train_y,validation_data= (valid_x,valid_y), epochs=epocs,batch_size=64 )


        ## Evaluate the model 
        eval_result = model.evaluate(valid_x,valid_y,batch_size= 64)

        eval_rmse = eval_result[1]

        # log the parameters and the resulsts 
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse",eval_rmse)

        # log the model 
        mlflow.tensorflow.log_model(model,"model",signature= signature)


        return {"loss":eval_rmse, "status":STATUS_OK, "model": model}
    


def objective(params):
    results = train_model(
        params,
        epocs= 3 ,
        train_x= train_x,
        train_y= train_y,
        valid_x= valid_x,
        valid_y= valid_y,
        test_x= test_x,
        test_y= test_y 
        )
    return results


space = { #basically the parameters we are going to try 
    "lr": hp.loguniform("lr",np.log(1e-5),np.log(1e-1)),
    "momentum":hp.uniform("momentum",0.0,1.0)
         }


mlflow.set_experiment("/wine-quality")
with mlflow.start_run():
    #counduct hyper parameter search  using hyperopt
    # traies is going to perform hypter paramerter tuining
    trails = Trials()
    best = fmin( # 

        fn = objective,
        space = space, 
        algo = tpe.suggest,
        max_evals=10,
        trials = trails
    )

    # featch detail for the best run 

    best_run = sorted(trails.results,key = lambda x:x["loss"])[0]

    #log the best parameters, loss and model 

    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"], "model", signature= signature)


    print("best parameter",best)
    print("best eval rmse",best_run["loss"])
    


