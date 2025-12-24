import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("data.csv")

    TARGET = "Attrition"

    drop_cols = [
        "id",               #индекс
        "EmployeeCount",     #всенда 1
        "Over18",            #всегда Y
        "StandardHours",     #всегда 80
        "Gender",            #почти не влияет 
        "PerformanceRating", #почти константа
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df[TARGET].astype(int)
    df = df.drop(columns=[TARGET])

    keep_num = [
        "Age",
        "MonthlyIncome",
        "JobLevel",
        "StockOptionLevel",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsWithCurrManager",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "RelationshipSatisfaction",
        "JobSatisfaction",
        "DistanceFromHome",
        "PercentSalaryHike",
        "NumCompaniesWorked",
    ]

    keep_num = [c for c in keep_num if c in df.columns]

    df_num = df[keep_num].copy()
 
    df_num["MonthlyIncome_log"] = np.log1p(df_num["MonthlyIncome"])
    df_num = df_num.drop(columns=["MonthlyIncome"])

    df_num["OverTime_bin"] = (df["OverTime"] == "Yes").astype(int)

    travel_map = {
        "Non-Travel": 0.0,
        "Travel_Rarely": 0.5,
        "Travel_Frequently": 1.0
    }
    df_num["BusinessTravel_code"] = df["BusinessTravel"].map(travel_map).fillna(0.0)
   
    marital_map = {
        "Divorced": 0.3,
        "Married": 0.5,
        "Single": 1.0
    }
    df_num["MaritalStatus_code"] = df["MaritalStatus"].map(marital_map).fillna(0.5)

    
    df_num["is_Department_HR"] = (df["Department"] == "Human Resources").astype(int)

    
    df_num["is_EduField_HR"] = (df["EducationField"] == "Human Resources").astype(int)
    df_num["is_EduField_Marketing"] = (df["EducationField"] == "Marketing").astype(int)

    
    df_num["is_SalesRepresentative"] = (df["JobRole"] == "Sales Representative").astype(int)
    df_num["is_JobRole_HR"] = (df["JobRole"] == "Human Resources").astype(int)
    df_num["is_LaboratoryTechnician"] = (df["JobRole"] == "Laboratory Technician").astype(int)
       
    df_num["is_ManufacturingDirector"] = (df["JobRole"] == "Manufacturing Director").astype(int)
    df_num["is_ResearchDirector"] = (df["JobRole"] == "Research Director").astype(int)

   
    if "YearsAtCompany" in df_num.columns and "TotalWorkingYears" in df_num.columns:
        df_num["years_in_company_share"] = df_num["YearsAtCompany"] / (df_num["TotalWorkingYears"] + 1.0)

    if "YearsWithCurrManager" in df_num.columns and "YearsAtCompany" in df_num.columns:
        df_num["years_with_manager_share"] = df_num["YearsWithCurrManager"] / (df_num["YearsAtCompany"] + 1.0)

 
    df_out = df_num.copy()
    df_out[TARGET] = y.values

    df_out = df_out.fillna(0)

    df_out.to_csv("data_f.csv", index=False)
    print(df_out.shape)
    
    

if __name__ == "__main__":
    main()
