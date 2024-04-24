import pandas as pd
from catboost import Pool
import joblib
import torch
import re

from llm import TransformerRegrModel


class VacancyAnalyzer:
    def __init__(self, transformer_path: str, catboost_path: str, inputs: dict):
        self.transformer_path = transformer_path
        self.catboost_path = catboost_path
        self.inputs = pd.DataFrame(inputs, index=[0]).drop(columns=['conversion', 'conversion_class', 'id'], axis=1)
        self.cat_features = ['profession', 'grade', 'location']
        self.text_features = ['emp_brand', 'mandatory', 'additional', 'comp_stages', 'work_conditions']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def __cleaner__(self, txt: str) -> str:
        txt = re.sub(r'\_(.*?)\_', r'', txt)
        txt = re.sub(r'([\n\t]*)', r'', txt)
        return txt

    def predict(self) -> float:
        df = self.inputs.drop(columns=self.text_features, axis=1)
        pool = Pool(df, cat_features=self.cat_features)
        regressor = joblib.load(self.catboost_path)
        prediction = regressor.predict(pool).tolist()
        return prediction[0]

    def classify(self) -> str:
        df = self.inputs[self.text_features]
        description = df[self.text_features[0]].values[0] + ' '
        for t in self.text_features[1:]:
            description += df[t].values[0]
            description += ' '
        description = self.__cleaner__(description)
        tbert = TransformerRegrModel('rubert', 3)
        tbert.load_state_dict(torch.load(self.transformer_path, map_location=torch.device(self.device)))
        tbert.to(self.device)
        tbert.eval()
        with torch.no_grad():
            outputs, _, _ = tbert(description)
            prediction = torch.argmax(outputs, 1).cpu().numpy()
        return prediction
