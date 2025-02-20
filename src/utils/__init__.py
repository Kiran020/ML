import os
import sys
import pickle
from src.exception import CustomException
import logging
from sklearn.metrics import r2_score  # ✅ Import r2_score

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():  # ✅ Correct iteration
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score  # ✅ Use model name as key
        
        return report

    except Exception as e:
        logging.info("Exception occurred while evaluating models")  # ✅ Corrected message
        raise CustomException(e, sys)
