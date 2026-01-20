# AI-ImpactSense

AI-ImpactSense is an earthquake risk prediction model that takes different earthquake parameters and predicts the risk level of that zone.

---

## Live Application
Deployed Web App:  
https://ai-impactsense-jghxyvcnxdftwkqjgr3imr.streamlit.app/

---

## Project Description
AI-ImpactSense is a machine learning–based web application developed to assess the potential impact of earthquakes using key seismic indicators.  
The system predicts the alert level of a region based on user-provided earthquake parameters, enabling early impact understanding and awareness.

---

## Input Parameters
The model uses the following seismic parameters for prediction:

| Parameter | Description |
|---------|------------|
| Magnitude | Earthquake strength |
| Depth | Depth of earthquake in kilometers |
| CDI | Community Determined Intensity |
| MMI | Modified Mercalli Intensity |
| SIG | Overall significance score |

---

## Output
- Predicted earthquake risk / alert level  
- Visual severity indication using color coding

---

## Machine Learning Model
- Algorithm: Random Forest Classifier  
- Training Data: Historical earthquake records  
- Stored Model Files:
  - `earthquake_impact_rf.pkl`
  - `feature_order.pkl`

---

## Technology Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

## Project Structure
```text
AI-impactSense/
 ├── app.py                     # Streamlit application
 ├── requirements.txt           # Project dependencies
 ├── earthquake_impact_rf.pkl   # Trained Random Forest model
 ├── feature_order.pkl          # Feature ordering for inference
 ├── README.md                  # Project documentation
 ├── Infosys_Springboard.ipynb  # Preprocessing notebook
 └── complete_AI_ImpactSense.ipynb  # Full training & analysis 
```
---

## Deployment
The application is deployed using Streamlit Community Cloud and can be accessed through the live application link provided above.

---

## Use Cases
- Earthquake impact awareness
- Educational demonstration of applied machine learning
- Academic project and evaluation

---

## Disclaimer
This project is intended strictly for educational purposes and should not be used as an official or real-time disaster warning system.


## Internship Context
This project was developed as part of the **Infosys Springboard Internship Program** as an academic and practical learning assignment focused on applying machine learning techniques to real-world problem statements.
