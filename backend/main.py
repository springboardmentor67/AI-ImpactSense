from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("model.pkl")    
# Load your existing dataset
df = pd.read_csv("earthquake_alert_balanced_dataset.csv")

# Train Decision Tree model once at startup
X = df[["magnitude", "depth", "cdi", "mmi", "sig"]]
y = df["alert"]
model = DecisionTreeClassifier()
model.fit(X, y)

# Input schema
class InputData(BaseModel):
    magnitude: float
    depth: float
    cdi: float
    mmi: float
    sig: float

@app.post("/predict")
def predict(data: InputData):
    input_vec = [[data.magnitude, data.depth, data.cdi, data.mmi, data.sig]]
    prediction = model.predict(input_vec)[0]
    return {"alert": prediction}

@app.get("/dataset")
def get_dataset_preview():
    return {"rows": df.to_dict(orient="records")}