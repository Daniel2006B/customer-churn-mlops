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
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── **init**.py
│   ├── data\_preparation.py
│   └── train.py
├── tests/
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
* tests — тести.

## Встановлення залежностей

Створення віртуального середовища у Linux/WSL:

python3 -m venv .venv-linux

Активація:

source .venv-linux/bin/activate

Встановлення залежностей:

python -m pip install -r requirements.txt

Валідація та версіонування даних

Перед навчанням моделі дані перевіряються за допомогою:

python src/validate\_data.py

Статус passed означає, що дані успішно пройшли перевірку.

Статус failed означає, що у даних знайдено помилки.

Підготовлена версія датасету зберігається у:

data/processed/churn\_v1.csv

Метадані версії зберігаються у:

data/processed/churn\_v1\_metadata.json

Для навчання модель використовує версію датасету v1.

Рекомендована послідовність запуску:

python src/validate\_data.py

python src/version\_data.py

python src/train.py

## Запуск навчання

python src/train.py

## Базова модель

Для прогнозування Customer Churn використовується модель
Logistic Regression.

Перед навчанням дані масштабуються за допомогою StandardScaler.
Модель і масштабування об'єднані в scikit-learn Pipeline.

## Отримані метрики

* Accuracy: 0.7637
* Precision: 0.7303
* Recall: 0.1738
* F1-score: 0.2808
* ROC-AUC: 0.8195

Після навчання модель зберігається у:

models/baseline\_model.joblib

Метрики зберігаються у:

reports/baseline\_metrics.json

## Висновок

У результаті виконання проєкту було створено базову структуру
MLOps-проєкту, підготовлено дані для задачі Customer Churn,
навчено модель Logistic Regression та оцінено її за основними
метриками класифікації.

Отримана модель є baseline-моделлю та може використовуватися
як початкова точка для подальшого покращення ML-системи.

## MLflow



Для відстеження ML-експериментів використовується MLflow.



Запуск MLflow UI:

mlflow ui --host 127.0.0.1 --port 5000



Після запуску MLflow UI доступний у браузері за адресою:

http://127.0.0.1:5000



Запуск навчання моделі:

python src/train.py



Для порівняння моделей виконуються окремі запуски:

python src/train.py --model-type logistic\_regression

python src/train.py --model-type random\_forest



MLflow зберігає інформацію про кожен запуск моделі:

параметри, метрики, версію даних, артефакти та модель.



MLflow runs зберігаються локально у папці:

mlruns/



У MLflow UI можна переглядати та порівнювати результати запусків.



У Model Registry зареєстрована модель:

CustomerChurnClassifier



Model Registry використовується для збереження та перегляду версій моделей.



Порівняння моделей



Було порівняно дві моделі:

Logistic Regression

Random Forest



Основною метрикою для вибору моделі обрано Recall.



Logistic Regression:

Recall: 0.5294



Random Forest:

Recall: 0.4759



Найкращою моделлю за метрикою Recall обрано Logistic Regression.

