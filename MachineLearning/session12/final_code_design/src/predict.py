import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ✅ البيانات — بدون Exited (ده target مش feature)
data = [
    [-0.7541830079917924,  0.5780143566720919,  0.11375998165198585, -0.14673040749854463, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    [-0.5605884106597949,  0.753908347743766,   0.7003528882054108,   1.6923927520037099,  0.0, 1.0, 0.0, 1.0, 9.0, 1.0],
    [ 0.11699268000219652, -0.3221490094005933,  0.5222180917013974, -0.8721429873346316,  1.0, 1.0, 0.0, 1.0, 5.0, 0.0],
    [ 0.6977764719981892,  -0.7256705183297281, -1.2170740485175422,  0.07677206232885857, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
]

columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender'
]

df = pd.DataFrame(data, columns=columns)

# ✅ لما run_train.py اشتغل، حفظ الموديل هنا مباشرة
model_path = r"C:\Users\Test\Desktop\DEPI-ONL4_AIS2_S2\MachineLearning\session12\final_code_design\outputs\3c454a76106e4804afe20a2961ed0d42\best_rf_model.joblib"

model = joblib.load(model_path)

preds = model.predict(df)
probs = model.predict_proba(df)[:, 1]

print("Predictions:", preds)
print("Probabilities:", probs)