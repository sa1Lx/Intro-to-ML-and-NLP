import pandas as pd
import numpy as np

names = ['Amit', 'Bina', 'Chirag', 'Deepa', 'Eshan', 'Farah', 'Gopal', 'Heena', 'Iqbal', 'Jaya']
subjects = ['Math', 'Science', 'History', 'Math', 'Science', 'History', 'Math', 'Science', 'History', 'Math']
scores = np.random.randint(50, 101, size=10)  # random int from 50 to 100

data = {
    'Name': names,
    'Subject': subjects,
    'Score': scores,
    'Grade': [''] * 10  # for 10 students
}

df = pd.DataFrame(data)

print(df)

for i in range(len(df)):
    score = df.loc[i, 'Score']
    
    if score >= 90:
        df.loc[i, 'Grade'] = 'A'
    elif score >= 80:
        df.loc[i, 'Grade'] = 'B'
    elif score >= 70:
        df.loc[i, 'Grade'] = 'C'
    elif score >= 60:
        df.loc[i, 'Grade'] = 'D'
    else:
        df.loc[i, 'Grade'] = 'F'

print(df)

sorted_df = df.sort_values(by='Score', ascending=False)

print(sorted_df)

for subject in df['Subject'].unique():
    scores = df[df['Subject'] == subject]['Score']
    avg = round(scores.mean(), 2)
    print(f"{subject}: {avg}")

def pandas_filter_pass(dataframe):
    result = dataframe[(dataframe['Grade'] == 'A') | (dataframe['Grade'] == 'B')]
    return result

# print(pandas_filter_pass(df))
