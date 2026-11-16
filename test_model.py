import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from predict import make_prediction
MODEL_PATH="best_model.joblib"
#FIXTURE 1:Load the trained model
 @pytest.fixture
 def model():
  assert os.path.exists(MODEL_PATH), "Model file not found!"
  return joblib.load(MODEL_PATH)
#FIXTURE 2:Prepare sample input data
 @pytest.fixture
 def sample_data():
  data=load_breast_cancer()
  X=pd.DataFrame(
  data.data,
  columns=data.feature_names
  )
  return X.iloc[:5] 
#TEST 1:Check whether model loads successfully
 def test_model_loads(model):
  assert model is not None
#TEST 2:Check whether model generates predictions
 def test_model_prediction(model,sample_data):
  predictions=model.predict(sample_data)
  assert prediction is not None
#TEST 3:Check prediction dimensions
 def test_prediction_shape(model,sample_data):
  predictions=model.predict(sample_data)
  assert len(predictions)==len(sample_data)
#TEST 4:Check valid prediction labels
 def test_prediction_values(model,sample_data):
  predictions=model.predict(sample_data)
  assert set(predictions).issubset({0,1})
#TEST 5:Check prediction probabilities
 def test_prediction_probabilities(model,sample_data):
   probabilities=model.predict_proba(sample_data)
   assert probabilities.shape==(5,2)
   assert np.all(probabilities>=0)
   assert np.all(probabilities<=1)
   assert np.allclose(probabilities.sum(axis=1),1)
#TEST 6:Check pipeline structure
  def test_model_pipeline(model):
   assert hasattr(model,"named_steps")
   assert "scaler" in model.named_steps
   assert "model" in model.named_steps
#TEST 7:Check reusable prediction function
 def test_make_prediction(model,sample_data):
   output=make_prediction(model,sample_data)
   assert "prediction" in output
   assert output["prediction"] in {"Malignant","Benign"}
#TEST 8:Check probabilities returned by prediction function
 def test_make_prediction_probabilities(model,sample_data)
 output=make_prediction(model,sample_data)
 probabilities=output["probabilities"] 
 assert "malignant" in probabilities
 assert "benign" in probabilities
 assert 0<=probabilities["malignant"]<=1
 assert 0<=probabilities["benign"]<=1

                
