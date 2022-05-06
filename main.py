import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



data = pd.read_excel('./Germandata.xlsx')

################### dropping unecessary columns

cols = ['Purpose','Age (Years)', 'Dependents','Telephone', 'Marital Status and Sex',   'Installment Plans',
'Housing', 'Job','Foreign Worker', 'Debtors / Guarantors','Residence (Years)','Property', 'Duration (months)']

data.drop(columns=cols, inplace=True)


############### Removing 'A' from columns to convert to numeric value for ML

data['Checking Account Status'] = data['Checking Account Status'].astype('string')
data['Saving Account / Bonds'] = data['Saving Account / Bonds'].astype('string')
data['Installment Rate'] = data['Installment Rate'].astype('string')
data['Credit History '] = data['Credit History '].astype('string')
data['Installment Rate'] = data['Installment Rate'].astype('string')



def remove_A(column_name):

    unique_val = data[column_name].unique()

    for i in unique_val:
        digit = i[-1]
        data[column_name].replace({i : digit}, inplace=True)


remove_A('Checking Account Status')
remove_A('Saving Account / Bonds')
remove_A('Credit History ')
remove_A('Employment (Years)')


data['Checking Account Status'] = data['Checking Account Status'].astype('int64')
data['Saving Account / Bonds'] = data['Saving Account / Bonds'].astype('int64')
data['Credit History '] = data['Credit History '].astype('int64')
data['Employment (Years)'] = data['Employment (Years)'].astype('int64')
data['Installment Rate'] = data['Installment Rate'].astype('int64')




#################### Removing outliers


def remove_outliers(columns):

    columns = list(columns)
    columns.remove('Cost Matrix')

    for col in columns:
            mean = data[col].mean()
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            S = 1.5*IQR
            LB = Q1 - S
            UB = Q3 + S
            data.loc[data[col] > UB,col] = np.nan
            data.loc[data[col] < LB,col] = np.nan
            data[col] = data[col].replace(np.nan, int(mean))

remove_outliers(data.columns)




####################### ML



X = data.drop(['Cost Matrix'], axis = 1)
y = data['Cost Matrix'].values

for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = i)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    print("Naive Bayes Accuracy : "+str(nb.score(X_test, y_test))+' '+str(i))




