# Customer Churn MLOps Project

## Мета проєкту

Метою проєкту є створення базового MLOps-процесу для прогнозування
відтоку клієнтів (Customer Churn).

На основі історичних даних про клієнтів модель визначає, чи може
клієнт припинити користуватися сервісом.

Задача належить до бінарної класифікації:

* 0 — клієнт залишився;
* 1 — клієнт пішов.

## Джерело даних

Для навчання моделі використано відкритий набір даних
Telco Customer Churn з Kaggle.

Дані зберігаються у:

data/raw/churn.csv

## Структура проєкту

customer-churn-mlops/
├── app/
│   ├── __init__.py
│   └── main.py
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── examples/
│   └── predict_request.json
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── train.py
│   ├── validate_data.py
│   └── version_data.py
├── tests/
├── Dockerfile
├── .dockerignore
├── README.md
├── requirements.txt
└── .gitignore

### Призначення основних директорій

* data/raw — початкові дані без змін;
* data/processed — підготовлені дані;
* src — код підготовки даних і навчання моделі;
* models — збережені ML-моделі;
* reports — метрики та результати;
* notebooks — експерименти та первинний аналіз;
* tests — тести;
* app — FastAPI inference-сервіс;
* examples — приклади запитів до API;
* configs — конфігурація валідації даних.

## Встановлення залежностей

Створення віртуального середовища у Linux/WSL:

python3 -m venv .venv-linux

Активація:

source .venv-linux/bin/activate

Встановлення залежностей:

python -m pip install -r requirements.txt

## Валідація та версіонування даних

Перед навчанням моделі дані перевіряються за допомогою:

python src/validate_data.py

Статус passed означає, що дані успішно пройшли перевірку.

Статус failed означає, що у даних знайдено помилки.

Підготовлена версія датасету зберігається у:

data/processed/churn_v1.csv

Метадані версії зберігаються у:

data/processed/churn_v1_metadata.json

Для навчання модель використовує версію датасету v1.

Рекомендована послідовність запуску:

python src/validate_data.py

python src/version_data.py

python src/train.py

## Запуск навчання

python src/train.py

Для запуску Logistic Regression:

python src/train.py --model-type logistic_regression

Для запуску Random Forest:

python src/train.py --model-type random_forest

## Базова модель

Для прогнозування Customer Churn використовується модель
Logistic Regression.

Також для порівняння використовується модель Random Forest.

Перед навчанням дані поділяються на тренувальну та тестову вибірки.

## Отримані метрики

Logistic Regression:

* Accuracy: 0.7935
* Precision: 0.6326
* Recall: 0.5294
* F1-score: 0.5764
* ROC-AUC: 0.8374

Random Forest:

* Accuracy: 0.8006
* Precision: 0.6768
* Recall: 0.4759
* F1-score: 0.5589
* ROC-AUC: 0.8335

Основною метрикою для вибору моделі обрано Recall.

Найкращою моделлю за метрикою Recall обрано Logistic Regression.

Остання модель зберігається у:

models/latest_model.joblib

Останні метрики зберігаються у:

reports/latest_metrics.json

## MLflow

Для відстеження ML-експериментів використовується MLflow.

Запуск MLflow UI:

mlflow ui --host 127.0.0.1 --port 5000

Після запуску MLflow UI доступний у браузері за адресою:

http://127.0.0.1:5000

Запуск навчання моделі:

python src/train.py

Для порівняння моделей виконуються окремі запуски:

python src/train.py --model-type logistic_regression

python src/train.py --model-type random_forest

MLflow зберігає інформацію про кожен запуск моделі:

параметри, метрики, версію даних, артефакти та модель.

MLflow runs зберігаються локально у папці:

mlruns/

У MLflow UI можна переглядати та порівнювати результати запусків.

У Model Registry зареєстрована модель:

CustomerChurnClassifier

Model Registry використовується для збереження та перегляду версій моделей.

## Порівняння моделей

Було порівняно дві моделі:

Logistic Regression

Random Forest

Основною метрикою для вибору моделі обрано Recall.

Logistic Regression:

Recall: 0.5294

Random Forest:

Recall: 0.4759

Найкращою моделлю за метрикою Recall обрано Logistic Regression.

## FastAPI

Для використання навченої моделі створено FastAPI inference-сервіс.

Файл сервісу:

app/main.py

Локальний запуск API:

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Після запуску API доступний за адресою:

http://127.0.0.1:8000

Перевірка стану сервісу:

curl.exe http://127.0.0.1:8000/health

Приклад відповіді:

{"status":"ok","model_loaded":true}

Endpoint /health використовується для перевірки, чи працює API та чи завантажена модель.

Swagger UI доступний за адресою:

http://127.0.0.1:8000/docs

У Swagger UI можна переглядати доступні endpoint-и та виконувати тестові запити.

Для прогнозування використовується endpoint:

POST /predict

Приклад JSON-запиту зберігається у:

examples/predict_request.json

Виконання запиту:

curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/predict_request.json"

Приклад відповіді:

{"churn_prediction":0,"churn_probability":0.28,"risk_label":"low"}

churn_prediction = 0 означає, що модель прогнозує відсутність відтоку клієнта.

churn_probability = 0.28 означає, що ймовірність відтоку становить 28%.

risk_label = low означає низький ризик відтоку.

## Обмеження inference

Під час навчання категоріальні ознаки перетворюються у числові колонки.

Під час inference набір колонок має відповідати набору ознак, на яких навчалася модель.

У поточній реалізації вхідні дані після pd.get_dummies() вирівнюються відповідно до колонок model.feature_names_in_.

Відсутні колонки заповнюються нульовими значеннями.

У реальній системі правильніше зберігати preprocessing разом із моделлю, наприклад за допомогою sklearn.pipeline.Pipeline.

## Docker

Для контейнеризації FastAPI-сервісу використовується Docker.

Збірка Docker-образу:

docker build -t churn-api:latest .

Запуск Docker-контейнера:

docker run --rm -p 8000:8000 churn-api:latest

Після запуску контейнера API доступний за адресою:

http://127.0.0.1:8000

Перевірка /health:

curl.exe http://127.0.0.1:8000/health

Отримана відповідь:

{"status":"ok","model_loaded":true}

Перевірка /predict:

curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@examples/predict_request.json"

Отримана відповідь:

{"churn_prediction":0,"churn_probability":0.28,"risk_label":"low"}

Docker дозволяє запускати API разом із кодом, залежностями та моделлю в ізольованому середовищі.

## Висновок

У результаті виконання проєкту було створено базовий MLOps-процес для прогнозування Customer Churn.

Було виконано підготовку, валідацію та версіонування даних, навчання і порівняння моделей, відстеження експериментів за допомогою MLflow, реєстрацію моделей у Model Registry, створення FastAPI inference-сервісу та його запуск у Docker-контейнері.

API дозволяє передавати дані клієнта та отримувати прогноз відтоку, його ймовірність і рівень ризику.