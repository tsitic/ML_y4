import pandas as pd
import numpy as np

feature_translation = {
    "id": "Идентификатор сотрудника",
    "Age": "Возраст",
    "BusinessTravel": "Частота командировок",
    "DailyRate": "Дневная ставка",
    "Department": "Отдел",
    "DistanceFromHome": "Расстояние от дома до работы",
    "Education": "Уровень образования",
    "EducationField": "Область образования",
    "EmployeeCount": "Количество сотрудников (служебный признак)",
    "EnvironmentSatisfaction": "Удовлетворённость рабочей средой",
    "Gender": "Пол",
    "HourlyRate": "Почасовая ставка",
    "JobInvolvement": "Вовлечённость в работу",
    "JobLevel": "Уровень должности",
    "JobRole": "Должность",
    "JobSatisfaction": "Удовлетворённость работой",
    "MaritalStatus": "Семейное положение",
    "MonthlyIncome": "Ежемесячный доход",
    "MonthlyRate": "Месячная ставка",
    "NumCompaniesWorked": "Количество предыдущих мест работы",
    "Over18": "Старше 18 лет",
    "OverTime": "Переработки",
    "PercentSalaryHike": "Процент повышения зарплаты",
    "PerformanceRating": "Оценка производительности",
    "RelationshipSatisfaction": "Удовлетворённость отношениями в коллективе",
    "StandardHours": "Стандартные рабочие часы",
    "StockOptionLevel": "Уровень опционов на акции",
    "TotalWorkingYears": "Общий стаж работы",
    "TrainingTimesLastYear": "Количество обучений за последний год",
    "WorkLifeBalance": "Баланс работа–жизнь",
    "YearsAtCompany": "Лет в компании",
    "YearsInCurrentRole": "Лет в текущей роли",
    "YearsSinceLastPromotion": "Лет с последнего повышения",
    "YearsWithCurrManager": "Лет с текущим руководителем",
    "Attrition": "Уход сотрудника (0 — остался, 1 — ушёл)"
}


df = pd.read_csv("data.csv")

TARGET = "Attrition"

print("ЦЕЛЕВАЯ ПЕРЕМЕННАЯ:", feature_translation[TARGET])
print("0 — сотрудник остался")
print("1 — сотрудник ушёл")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True))


numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET)
categorical_cols = df.select_dtypes(exclude=[np.number]).columns


print("ЧИСЛОВЫЕ ПРИЗНАКИ")

for col in numeric_cols:
    print("-" * 100)
    print(f"ПРИЗНАК: {col}")
    print(f"Описание: {feature_translation.get(col, 'Нет перевода')}")
    print(df.groupby(TARGET)[col].describe())
    print("-" * 100, "\n")

print("КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ")

for col in categorical_cols:
    print("-" * 100)
    print(f"ПРИЗНАК: {col}")
    print(f"Описание: {feature_translation.get(col, 'Нет перевода')}")
    print(pd.crosstab(df[col], df[TARGET], normalize="index"))
    print("-" * 100, "\n")
    