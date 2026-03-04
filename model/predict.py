import pickle
import pandas as pd

# import the ml model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

class_labels = model.classes_

def predict_output(user_input : dict):
    df = pd.DataFrame([user_input])

    
    predicted_class = model.predict(df)[0]
    
    probablities = model.predict_proba(df)[0]
    confidence = max(probablities)
    
    class_probs = dict(zip(class_labels, map(lambda p: round(p,4), probablities)))
    
    return {
        "predicted_category":predicted_class,
        "confidence":round(confidence, 4),
        "class_probablities": class_probs
    }    