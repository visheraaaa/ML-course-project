import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import random as rnd


apps = pd.read_csv("data (7).csv")
# print(apps.head(10))

pd.set_option('display.max_columns', None)

apps['days_exposition'] = apps['days_exposition'].fillna(0).astype('int')
apps['is_apartment'] = apps['is_apartment'].fillna("False").astype('string')
apps = apps.loc[apps['locality_name'].isnull() == False]
apps = apps.loc[apps['floors_total'].isnull() == False]
apps['balcony'] = apps['balcony'].fillna(0).astype('int')
apps['price_for_m2'] = (apps['last_price'] / apps['total_area']).astype('int')

def fillna_ceiling_height(row):
    if row['cityCenters_nearest'] < 2500:
        return 'самый центр'
    elif row['cityCenters_nearest'] < 5000:
        return 'центр'
    elif row['cityCenters_nearest'] < 17000:
        return 'в пределах города'
    elif row['cityCenters_nearest'] >= 17000:
        return 'область'

apps['fromcenter_category'] = apps.apply(fillna_ceiling_height, axis=1)


def fillna_ceiling(row):
    if pd.isna(row["ceiling_height"]):
        if row['fromcenter_category'] == "самый центр":
             return 3.24
        elif row['fromcenter_category'] == "центр":
            return 3.19
        elif row['fromcenter_category'] == "в пределах города":
            return 2.75
        elif row['fromcenter_category'] == "область":
            return 2.68
    return row["ceiling_height"]

def fillna_parks(row):
    if row["parks_around3000"] == 0 and pd.isna(row["parks_nearest"]):
        return 2
    elif pd.isna(row['parks_nearest']) == False and row['parks_nearest'] < 300:
        return 0
    elif pd.isna(row['parks_nearest']) == False and row['parks_nearest'] >= 300 and row['parks_nearest'] <= 700:
        return 1
    elif pd.isna(row['parks_nearest']) == False and row['parks_nearest'] > 700:
        return 2
    return row['parks_nearest']

def fillna_ponds(row):
    if row["ponds_around3000"] == 0 and pd.isna(row["ponds_nearest"]):
        return 2
    elif pd.isna(row['ponds_nearest']) == False and row['ponds_nearest'] < 300:
        return 0
    elif pd.isna(row['ponds_nearest']) == False and row['ponds_nearest'] >= 300 and row['ponds_nearest'] <= 700:
        return 1
    elif pd.isna(row['ponds_nearest']) == False and row['ponds_nearest'] > 700:
        return 2
    return row['ponds_nearest']
def string_to_bool(value):
    return value == "True"

def fill_distance(value):
    if value < 300:
        return "Близко"
    elif value > 300 and value < 700:
        return "Средне"
    else:
        return "Далеко"

apps["ceiling_height"] = apps.apply(fillna_ceiling, axis=1)

apps["parks_nearest"] = apps.apply(fillna_parks, axis=1)
apps["ponds_nearest"] = apps.apply(fillna_ponds, axis=1)



# print(apps.head(10))
dropped = apps.dropna()
le = LabelEncoder()
# dropped = dropped[["total_area", "rooms", "fromcenter_category", "last_price"]]
dropped = pd.get_dummies(dropped, columns=["fromcenter_category"])

dropped = dropped.drop('first_day_exposition', axis=1)
dropped = dropped.drop('locality_name', axis=1)
dropped = dropped.drop('id', axis=1)
dropped = dropped.drop('total_images', axis=1)

category_counts = dropped['ponds_nearest'].value_counts()



dropped["is_apartment"] = dropped["is_apartment"].apply(string_to_bool)
dropped["studio"] = dropped["is_apartment"].apply(string_to_bool)
dropped["open_plan"] = dropped["is_apartment"].apply(string_to_bool)


# dropped = dropped.loc[dropped['fromcenter_category_центр']==True]

x = dropped.drop(["last_price", 'price_for_m2'], axis=1).values
y = dropped["last_price"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


index = rnd.randint(0, 1000)

sample = dropped.iloc[index]

# Признаки для предсказания
sample_features = sample.drop(['last_price', 'price_for_m2']).values.reshape(1, -1)

# Реальное значение
actual_value = sample['last_price']

# Предсказанное значение
predicted_value = model.predict(sample_features)[0]

print(f"Реальное значение: {actual_value}")
print(f"Предсказанное значение: {predicted_value}")

print(dropped[dropped['fromcenter_category_область'] == True].shape[0])


dropped.plot(kind = 'scatter', y = 'price_for_m2', x = 'cityCenters_nearest', alpha = 0.8)


dropped['cityCenters_nearest'].corr(dropped['price_for_m2'])





def predict_price(row):
    df = pd.read_csv('final_dropped.csv')
    x = df.drop(["last_price", "Unnamed: 0"], axis=1).values
    y = df["last_price"].values
    x_columns = df.drop(["last_price"], axis=1).columns
    args = row.split(",")
    counter = 1
    line = []
    for u in args:
        if u == "False":
            line.append(False)
        elif u == "True":
            line.append(True)
        else:
            line.append(float(u))
        counter += 1
    new_df = pd.DataFrame(line).values.reshape(1, -1)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_tr, y_tr)
    return model.predict(new_df)






# print(dropped.shape[0])
# dropped.to_csv("final_dropped.csv", encoding='utf-8')


# plt.boxplot(apps[apps['days_exposition']!=0]['days_exposition'])
# plt.ylim(1,1000)
#
# apps.plot(y = 'days_exposition', kind = 'hist', bins = 30, grid = True, range = (1,1600))
# apps.plot(y = 'days_exposition', kind = 'hist', bins = 100, grid = True, range = (1,200))
#
# #среднее значение, медиана и межквартильный размах
# apps[apps['days_exposition']!=0]['days_exposition'].describe()
# plt.show()
