import numpy as np
import pandas as pd
import joblib
from pipeline import input_func, helper_func


# def input_func():
#     Age = float(input('Enter age: '))
#     Debt = float(input('Enter debt: '))
#     BankCustomer = int(input('Bank customer ? (0 or 1): '))
#     Industry = input('Enter Industry: ')
#     YearsEmployed = float(input('Number of years of employment: '))
#     PriorDefault = int(input('Prior Defaulter ? (0 or 1): '))
#     Employed = int(input('Employed ? (0 or 1): '))
#     CreditScore = float(input('Enter your Credit Score: '))
#     DriversLicense = int(input('Drivers License ? (0 or 1): '))
#     Income = float(input('Enter your Income: '))
    
#     df = pd.DataFrame({'Age':[Age], 'Debt': [Debt], 'BankCustomer': [BankCustomer], 'Industry':[Industry], 'YearsEmployed':[YearsEmployed], 'PriorDefault':[PriorDefault], 'Employed':[Employed], 'CreditScore':[CreditScore], 'DriversLicense':[DriversLicense], 'Income':[Income]})
    
#     return df




# def helper_func(df):
#     for idx, row in df.iterrows():
#         if df.loc[idx, 'Age']< 30:
#             df.loc[idx, 'Age'] = 'youth'

#         elif df.loc[idx, 'Age']>=30 and df.loc[idx, 'Age']<40:
#             df.loc[idx, 'Age'] = 'youthWithResponsibility'

#         elif df.loc[idx, 'Age']>=40 and df.loc[idx, 'Age']<55:
#             df.loc[idx, 'Age'] = 'midLife'

#         elif df.loc[idx, 'Age']>=55:
#             df.loc[idx, 'Age'] = 'towardsRetirement'


#     for idx, row in df.iterrows():

#         if df.loc[idx, 'Debt']>=0 and df.loc[idx, 'Debt']<5:
#             df.loc[idx, 'Debt'] = 'Green'

#         elif df.loc[idx, 'Debt']>=5 and df.loc[idx, 'Debt']<10:
#             df.loc[idx, 'Debt'] = 'Yellow'

#         elif df.loc[idx, 'Debt']>=10 and df.loc[idx, 'Debt']<15:
#             df.loc[idx, 'Debt'] = 'Orange'

#         elif df.loc[idx, 'Debt']>=15:
#             df.loc[idx, 'Debt'] = 'Red'

#     for idx, row in df.iterrows():

#         if df.loc[idx, 'CreditScore']>=0 and df.loc[idx, 'CreditScore']<5:
#             df.loc[idx, 'CreditScore'] = 'Red'

#         elif df.loc[idx, 'CreditScore']>=5 and df.loc[idx, 'CreditScore']<10:
#             df.loc[idx, 'CreditScore'] = 'Yellow'

#         elif df.loc[idx, 'CreditScore']>=10:
#             df.loc[idx, 'CreditScore'] = 'Green'

#     for idx, row in df.iterrows():

#         if df.loc[idx, 'YearsEmployed']>=0 and df.loc[idx, 'YearsEmployed']<0.5:
#             df.loc[idx, 'YearsEmployed'] = 'Fresher'

#         elif df.loc[idx, 'YearsEmployed']>=0.5 and df.loc[idx, 'YearsEmployed']<2.56:
#             df.loc[idx, 'YearsEmployed'] = 'Experienced'

#         elif df.loc[idx, 'YearsEmployed']>=2.56:
#             df.loc[idx, 'YearsEmployed'] = 'Expert'


#     for idx, row in df.iterrows():

#         if df.loc[idx, 'Income']>=0.0 and df.loc[idx, 'Income']<51.8:
#             df.loc[idx, 'Income'] = 'Poor'

#         elif df.loc[idx, 'Income']>=51.8 and df.loc[idx, 'Income']<2000.0:
#             df.loc[idx, 'Income'] = 'Middle Class'

#         elif df.loc[idx, 'Income']>=2000.0:
#             df.loc[idx, 'Income'] = 'Rich'


#     for idx, row in df.iterrows():
#         if df.loc[idx, 'Industry'] in ['Utilities', 'Real Estate', 'Education', 'Research', 'Transport']:
#             df.loc[idx, 'Industry'] = 'Others'
            
    
#     return df


transformer = joblib.load('transformer.sav')
pca = joblib.load('pca.sav')
model = joblib.load('model.sav')

df = input_func()
df = helper_func(df)

df = transformer.transform(df)
df = pca.transform(df)
ypred = model.predict(df)

print(ypred[0])