from fastapi import APIRouter, HTTPException, status
from models import Prediction_Input
from models import Prediction_Output
import pandas as pd
import pickle

MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
PROCESS_PATH = 'model/PII_model.pickle'
MODEL_PATH =  "model/xgb_reg.pkl"
# Load Tensorflow model
selected_var = ['EK', 'Mean_Integrated', 'Skewness', 'imp', 'SD_DMSNR_Curve', 'Mean_DMSNR_Curve']
model = pickle.load(open(MODEL_PATH, "rb"))
preprocessor_loaded = pickle.load(open(PROCESS_PATH, "rb"))

#print(model.summary())

router = APIRouter()

preds = []

def FE(df):
    df['Z'] = (df["SD"]*df["SD"] - df["Mean_Integrated"])/df["SD"]
    df['imp'] = (df["EK"]*10)*(df["EK"])+ df['SD_DMSNR_Curve']
    return df

@router.get("/ml")
def get_preds():
    return preds

@router.post('/ml', status_code=status.HTTP_201_CREATED)
def create_pred(pred_input : Prediction_Input):
    X = preprocessor_loaded.transform(FE(pd.DataFrame([[pred_input.EK,pred_input.SD,pred_input.Mean_Integrated,pred_input.Skewness,pred_input.SD_DMSNR_Curve,pred_input.Mean_DMSNR_Curve]],columns = ['EK','SD','Mean_Integrated','Skewness','SD_DMSNR_Curve','Mean_DMSNR_Curve'])).loc[:, selected_var])
    print(X)
    y_pred = model.predict(X)
    prediction_dict = {"id": str(len(preds)), "pred" : float(y_pred[0] )}
    preds.append(prediction_dict)
    return {"message": "Creado satisfactoriamente"}

