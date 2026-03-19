import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore") 
data = [
 [-0.7541830079917924, 0.5780143566720919, 0.11375998165198585, -0.14673040749854463, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0],
 [-0.5605884106597949, 0.753908347743766, 0.7003528882054108, 1.6923927520037099, 0.0, 1.0, 0.0, 1.0, 9.0, 1.0, 1.0],
 [0.11699268000219652, -0.3221490094005933, 0.5222180917013974, -0.8721429873346316, 1.0, 1.0, 0.0, 1.0, 5.0, 0.0, 2.0],
 [0.6977764719981892, -0.7256705183297281, -1.2170740485175422, 0.07677206232885857, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0]
]
columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography',
    'Gender', 'Exited'
]
# df = pd.DataFrame(data, columns=columns)
# model_path = r'D:/InnovionRay_team/Innovisionray_team/InnoVisionRay/Amit/AI_Diploma/Data_Science/Machine Learning/MachineLearning/MLops/MlFlow/MLFlowProject/MLFlowProject/P1/outputs/b41585bae46f49148d099f7589b4acbf/artifacts/model_rf/model.pkl'
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)
# preds = model.predict(df)
# probs = model.predict_proba(df)
# print("Predictions:", preds)
# print("Probabilities:", probs)
import mlflow
mlflow.set_tracking_uri("file:///C:/Users/Test/Desktop/DEPI-ONL4_AIS2_S2/MachineLearning/session12/final_code_design/mlruns")
model_uri = 'runs:/f009b5d7f2a64a599eaa86f794483c3b/model_rf'
# Replace INPUT_EXAMPLE with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
input_data = data
# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
 model_uri=model_uri,
 input_data=input_data,
 env_manager="uv",
)
# file:d:/InnovionRay_team/Innovisionray_team/InnoVisionRay\Amit\AI_Diploma\Data_Science\Machine Learning\MachineLearning\MLops\MlFlow\MLFlowProject\MLFlowProject\P1\outputs\mlruns/321286496620637019/b41585bae46f49148d099f7589b4acbf/artifacts/model_rf
import mlflow
logged_model = 'file:///C:/Users/Test/Desktop/DEPI-ONL4_AIS2_S2/MachineLearning/session12/final_code_design/mlruns/321286496620637019/b41585bae46f49148d099f7589b4acbf/artifacts/model_rf'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data)) 
