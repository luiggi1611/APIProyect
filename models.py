from pydantic import BaseModel

class Todo(BaseModel):
    id: int
    task : str
    completed : bool

class Prediction_Input(BaseModel):
    EK : float
    Mean_Integrated : float
    Skewness : float
    SD : float
    SD_DMSNR_Curve : float
    Mean_DMSNR_Curve : float

class Prediction_Output(BaseModel):
    id : int
    pred : int



    
