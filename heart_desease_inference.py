import joblib
import pandas as pd

# Загрузка модели XGBoost и масштабатора
best_model = joblib.load('neural_network_model.joblib')
best_scaler = joblib.load('scaler.joblib')

# Загрузка тестового набора данных
test = pd.read_csv('test2.csv')

# Сохранение ID из тестового набора
test_ids = test['ID']

# Удаление столбца 'ID' из данных перед предобработкой
test_processed = test.drop(['ID'], axis=1)

# Преобразование категориальных признаков
categorical_variables = ['slope', 'sex', 'number_of_major_vessels', 'resting_electrocardiographic_results', 'fasting_blood_sugar', 'thal', 'exercise_induced_angina']

# Преобразование категориальных переменных с использованием One-Hot Encoding
encoder2 = OneHotEncoder(drop='first', sparse_output=False)
encoded_features2 = encoder2.fit_transform(test[categorical_variables])

# Создание DataFrame из закодированных переменных
encoded_df2 = pd.DataFrame(encoded_features2, columns=encoder2.get_feature_names_out(categorical_variables))

# Сброс индексов для корректного объединения
encoded_df2.reset_index(drop=True, inplace=True)

# Объединение закодированной переменной с исходными данными
test1 = pd.concat([test.reset_index(drop=True), encoded_df2], axis=1)

# Удаляем исходные категориальные переменные
test1.drop(categorical_variables, axis=1, inplace=True)

# Приведение тестовых данных к той же структуре, что и обучающие данные
# Определение признаков, которые использовались при обучении скэйлера 
expected_features = best_scaler.feature_names_in_

# Сортировка признаков в тестовом наборе данных в том же порядке, как и в обучающих данных
test1_processed = test1.reindex(columns=expected_features, fill_value=0)

# Масштабирование числовых признаков
test1_processed_scaled = best_scaler.transform(test1_processed)

# Предсказания на тестовом наборе
test_predictions_proba = best_model.predict(test1_processed_scaled).ravel()
test_predictions = (test_predictions_proba > 0.5).astype(int)

# Создание DataFrame для сохранения результатов
predicted = pd.DataFrame({
    'ID': test_ids,
    'class': test_predictions
})

# Сохранение результатов
predicted.to_csv('submission.csv', index=False)
print("Инференс завершён. Результаты сохранены в 'predicted.csv'.")