import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score



df=pd.read_csv("salary_prediction/student_placement_prediction_dataset_2026.csv")
#  print(df.head(5))

#  print(df.isnull().sum())

# print(df.info())

# print(df.describe())

#  print(df.duplicated().sum())

#  """finding categorical col in your dataset"""
# cat_col=df.select_dtypes(include='object')
# print(cat_col.columns)


# """finding the numercal columns"""
# num_cols=df.select_dtypes(include="number")
# print(num_cols.columns)


# """finding the corelation in numeric data only"""
# print(df.corr(numeric_only=True)['salary_package_lpa'])

# sns.histplot(df['salary_package_lpa'],kde=True)
# plt.grid(True)

# plt.show()


# print(df["salary_package_lpa"].min())
# print(df["salary_package_lpa"].max())
# print(df["salary_package_lpa"].value_counts())
# print(df["salary_package_lpa"].max())


# """to ckeck the skeweness"""
# print(df['salary_package_lpa'].skew())
# """the salary is left skwed because around 25% of the student do not got placement 
# due to which it cause left skwe"""


'''below one give you the exact number of the std who is not placed or placed'''
# print((df['salary_package_lpa']==0).sum())
# print((df['salary_package_lpa']>0).sum())

'''histplot on cgpa'''

# sns.histplot(df['cgpa'],kde=True)
# plt.show()
# print(df['cgpa'].skew()) 
# print((df['cgpa']<10).sum())
# print((df['cgpa']==10).sum())

"""count plot on categorical data"""
# cols=['gender','branch','college_tier','volunteer_experience','placement_status']

# for col in cols:
#     plt.figure(figsize=(6,4))
#     sns.countplot(data=df,x=col)
#     plt.title(col)
#     plt.show()
# ''''''
# print(df.sample(5))
"""to count the number of person in differnt section of the dataset"""

# print(df['gender'].value_counts())
# print(df['placement_status'].value_counts())

# print(df['branch'].value_counts())
# print(df['college_tier'].value_counts())
# print(df['volunteer_experience'].value_counts())
# print(pd.crosstab(df['branch'],df['gender']))

"""plot for getting the corr"""
# plt.figure(figsize=(14,10))
# sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",fmt='.2f')
"""it give the proper heatmap where annot shows the corr value in your heatma"""
# plt.show()
"""scatter plot"""
# sns.scatterplot(data=df,x='cgpa',y='salary_package_lpa',hue='placement_status')
# plt.show()
"""plot for categical vs numerical"""

# # sns.barplot(data=df,x='branch',y='salary_package_lpa',hue='placement_status')
# # plt.show()
# print(df.groupby(['branch', 'placement_status'])['salary_package_lpa'].mean())

"""box plot """

# sns.boxplot(data=df,y='cgpa')
# plt.show()
'''checking the outlier'''
Q1=df['cgpa'].quantile(0.25)
Q3=df['cgpa'].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

# print('lowe:',lower)
# print('upper',upper)

outlier=df[(df['cgpa']<lower) | (df['cgpa']>upper)]
# print("number of outlier",len(outlier))
num_cols=df.select_dtypes(include="number")
skewww=num_cols.skew()
print(skewww.sort_values(ascending=False))

df.drop(columns=['student_id','placement_status'],inplace=True)
# print(df.info())
x_train,x_test,y_train,y_test=train_test_split(df.drop(['salary_package_lpa'],axis=1),df['salary_package_lpa'],test_size=0.2,random_state=42)
# print(x_train)

processor=ColumnTransformer([
        ('branch',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),['branch']),
        ('gender',OneHotEncoder(drop='if_binary',handle_unknown='ignore'),['gender']),
        ('volunteer_experience',OneHotEncoder(drop='if_binary',handle_unknown="ignore"),['volunteer_experience']),


        ('college_tier',OrdinalEncoder(categories=[['Tier 3','Tier 2','Tier 1']]),['college_tier']),


    
],remainder='passthrough'
)
poly=PolynomialFeatures(degree=2,include_bias=False)




pipe=Pipeline([
    ('processor',processor),
    ('poly',PolynomialFeatures(degree=2,include_bias=False)),
    # ('model',LinearRegression())
    ('model', RandomForestRegressor())
 
])

pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

print(y_pred)
result=pd.DataFrame(
    {
        'actual':y_test.values,
        'predicted':y_pred
    }
)
print(result.head(10))

print("r2 score",r2_score(y_test,y_pred))
print("mse", mean_squared_error(y_test,y_pred))
print("rmse", np.sqrt(mean_squared_error(y_test,y_pred)))

# # # print(df.corr(numeric_only=True)['salary_package_lpa'].sort_values(ascending=False))
# # """console app"""

# age=int(input("enter the age: "))
# gender=input("enter the gender(male/female): ")
# cgpa=float(input("enter the cgpa: "))
# branch=input("enter the branch: ")
# college_tier=input("enter the college tier(tier1/tier2/tier3): ")
# internships=int(input("enter the number of internships: "))
# projects=int(input("enter the no of project: "))
# certifications=int(input("number of certification: "))
# coding=float(input("coding score(0-100): "))
# aptitude=float(input("enter the amplitude score(0-100): "))
# communication = float(input("Communication skill score (0-100): "))
# logical = float(input("Logical reasoning score (0-100): "))
# hackathons = int(input("Hackathons participated: "))
# github = int(input("GitHub repositories: "))
# linkedin = int(input("LinkedIn connections: "))
# mock = float(input("Mock interview score (0-100): "))
# attendance = float(input("Attendance percentage: "))
# backlogs = int(input("Number of backlogs: "))
# extra = float(input("Extracurricular score (0-100): "))
# leadership = float(input("Leadership score (0-100): "))
# volunteer = input("Volunteer experience (Yes/No): ")
# sleep = float(input("Sleep hours per day: "))
# study = float(input("Study hours per day: "))

# new_student=pd.DataFrame(
#     {
#         "age":[age],
#         'gender':[gender],
#             "cgpa": [cgpa],
#     "branch": [branch],
#     "college_tier": [college_tier],
#     "internships_count": [internships],
#     "projects_count": [projects],
#     "certifications_count": [certifications],
#     "coding_skill_score": [coding],
#     "aptitude_score": [aptitude],
#     "communication_skill_score": [communication],
#     "logical_reasoning_score": [logical],
#     "hackathons_participated": [hackathons],
#     "github_repos": [github],
#     "linkedin_connections": [linkedin],
#     "mock_interview_score": [mock],
#     "attendance_percentage": [attendance],
#     "backlogs": [backlogs],
#     "extracurricular_score": [extra],
#     "leadership_score": [leadership],
#     "volunteer_experience": [volunteer],
#     "sleep_hours": [sleep],
#     "study_hours_per_day": [study]

#     }
# )

# prediction=pipe.predict(new_student)
# print("\n=========================")
# print(f"The predictied salary package is {prediction[0]}LPA")
# print("==========================")

'''"The Linear Regression model achieved an R² score of approximately 0.04. 
Correlation analysis showed that most features have very weak linear relationships with salary.
 Therefore, the poor performance is likely due to limited predictive 
information in the dataset or the need for a more complex model.'''