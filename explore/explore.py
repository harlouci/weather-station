import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.weather.data.prep_datasets import make_dataset
from src.weather.data.data_transformers import make_target_transformer, TargetChoice, make_cleaning_transformer, make_remove_last_rows_transformer
from src.weather.features.skl_features import FeatureNames, make_input_transformer
from src.weather.models.models import models
from src.weather.metrics.evaluate import score_evaluation



df = pd.read_csv('weather_dataset_raw.csv')

target_choice = TargetChoice('Weather_conditions', 4)
feature_names = FeatureNames(numerical=['Temperature_C', 'Humidity', 'Wind_speed_kmph', 'Wind_bearing_degrees', 'Visibility_km', 'Pressure_millibars'],
                             categorical=['Weather_conditions', 'Month'])

cleaning_transformer = make_cleaning_transformer(target_choice)
target_tranformer = make_target_transformer(target_choice)
input_transformer = make_input_transformer(feature_names, target_choice)
remove_last_rows_transformer = make_remove_last_rows_transformer(target_choice)




df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
df.sort_values(by='Timestamp', ascending=True, inplace=True)

dataset = make_dataset(data=df,
                        training_transform = cleaning_transformer,
                        test_transform=cleaning_transformer,
                        target_transform=target_tranformer,
                       remove_last_rows_transformer = remove_last_rows_transformer)

print('percentage rain', dataset.train_y.mean())

models_score = {}
best_valid_score = 0
best_valid_model = None

for model_name, model_config in models.items():
    model = model_config['model']
    param_grid = model_config['param_grid']

    model_pipeline = Pipeline([
            ('preprocessor', input_transformer),
            ('model', model)
        ])

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5)


    grid_search.fit(dataset.train_x, dataset.train_y)

    best_model = grid_search.best_estimator_

    best_model.fit(dataset.train_x, dataset.train_y)

    y_pred = best_model.predict(dataset.train_x)



    accuracy = score_evaluation(accuracy_score, best_model, dataset)

    models_score[model_name] = {'model':best_model, 'score':accuracy}
    print(model)
    print(accuracy)

for model_name, model_score in models_score.items():
    valid_score = model_score['score'].valid
    if valid_score>best_valid_score:
        best_valid_score=valid_score
        best_model = model_score['model']

joblib.dump(best_model, 'best_model.pkl')


# Load the saved model from the file
new_df = pd.read_csv('weather_dataset_raw.csv')
loaded_model = joblib.load('best_model.pkl')
new_pred = loaded_model.predict(new_df)
print(new_pred)






