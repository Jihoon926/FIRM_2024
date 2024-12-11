import numpy as np
import pandas as pd
from datetime import datetime

# 한국 무위험 수익률은 한국은행 ecos에서, 미국 무위험 수익률은 investing.com에서 가져옴.

def get_risk_free_df():
    domestic_rf = pd.read_csv("Valuation/Risk_free/korean_rf.csv")
    domestic_rf = domestic_rf.transpose().reset_index()
    domestic_rf.columns = ['Date', 'Price']
    datetime_format_domestic = "%Y/%m/%d"

    foreign_rf = pd.read_csv("Valuation/Risk_free/foreign_rf.csv")[['Date', 'Price']]
    datetime_format_foreign = "%m/%d/%Y"

    for i in range(len(domestic_rf)):
        domestic_rf.loc[i, 'Date'] = datetime.strptime(domestic_rf.loc[i, 'Date'], datetime_format_domestic).date()

    for i in range(len(foreign_rf)):
        foreign_rf.loc[i, 'Date'] = datetime.strptime(foreign_rf.loc[i, 'Date'], datetime_format_foreign).date()

    domestic_workday = domestic_rf['Date']
    foreign_workday = foreign_rf['Date']

    domestic_foreign = list(set(domestic_workday) - set(foreign_workday))
    foreign_domestic = list(set(foreign_workday) - set(domestic_workday))
    for day in domestic_foreign:
        domestic_rf = domestic_rf.drop(domestic_rf[domestic_rf['Date'] == day].index)

    for day in foreign_domestic:
        foreign_rf = foreign_rf.drop(foreign_rf[foreign_rf['Date'] == day].index)

    domestic_rf.reset_index(inplace = True, drop = True)
    foreign_rf.sort_values(by=['Date'], inplace = True)
    foreign_rf.reset_index(inplace = True, drop = True)
    return domestic_rf, foreign_rf

def get_diff():
    domestic_rf = pd.read_csv("Valuation/Risk_free/korean_rf.csv")
    domestic_rf = domestic_rf.transpose().reset_index()
    domestic_rf.columns = ['Date', 'Price']
    datetime_format_domestic = "%Y/%m/%d"

    foreign_rf = pd.read_csv("Valuation/Risk_free/foreign_rf.csv")[['Date', 'Price']]
    datetime_format_foreign = "%m/%d/%Y"

    for i in range(len(domestic_rf)):
        domestic_rf.loc[i, 'Date'] = datetime.strptime(domestic_rf.loc[i, 'Date'], datetime_format_domestic).date()

    for i in range(len(foreign_rf)):
        foreign_rf.loc[i, 'Date'] = datetime.strptime(foreign_rf.loc[i, 'Date'], datetime_format_foreign).date()

    domestic_workday = domestic_rf['Date']
    foreign_workday = foreign_rf['Date']

    domestic_foreign = list(set(domestic_workday) - set(foreign_workday))
    foreign_domestic = list(set(foreign_workday) - set(domestic_workday))
    for day in domestic_foreign:
        domestic_rf = domestic_rf.drop(domestic_rf[domestic_rf['Date'] == day].index)

    for day in foreign_domestic:
        foreign_rf = foreign_rf.drop(foreign_rf[foreign_rf['Date'] == day].index)

    domestic_rf.reset_index(inplace = True, drop = True)
    foreign_rf.sort_values(by=['Date'], inplace = True)
    foreign_rf.reset_index(inplace = True, drop = True)
    rate_diff = domestic_rf.sub(foreign_rf)
    rate_diff['Date'] = domestic_rf['Date']
    return rate_diff
