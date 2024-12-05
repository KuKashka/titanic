import pandas as pd
df = pd.read_csv('titanic.csv')
print(df.info())

df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
df.info()

def sex_clean(data):
    if data == "male":
        return 1
    else:
        return 0 

df['Sex'] = df['Sex'].apply(sex_clean)
df.info()

df["Embarked"] = df['Embarked'].fillna("S")
df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])
df = df.drop(["Embarked"], axis=1)

age = df["Age"].mean()
df["Age"] = df["Age"].fillna(round(age))
df.info()

df.to_csv("clean_titanic.csv", index=False)
