import telebot
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def predict_price(row):
    df = pd.read_csv('final_dropped.csv')
    x = df.drop(["last_price", "Unnamed: 0", 'price_for_m2'], axis=1).values
    y = df["last_price"].values
    args = row.split("\n")
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

token = "7056454152:AAHJn2jKSILVC04yY2Du-m9HZOSbK49TfsM"
bot = telebot.TeleBot(token)
hellomessage = "Привет! Я бот, оценивающий квартиры. Чтобы я смог оценить квартиру, введи параметры через Enter: \n Метраж квартиры \n Количество комнат \n Высота потолка \n Этажность дома \n Площадь жилой зоны \n Этаж квартиры \n Является ли квартира апартаментами? \n Студией? \n Имеет ли жильё открытую планировку? \n Площадь кухни \n Количество балконов \n Расстояние до ближайшего аэропорта \n  Расстояние до ближайшего центра города \n Количество парков в радиусе 3000 метров \n Расстояние до ближайшего парка \n Количество прудов в радиусе 3000 метров \n Расстояние до ближайшего пруда\n Сколько дней висит объявление (по умолчанию 90) \n Находится ли кваритра в пределах города, в области, в центре или в самом центре? (пример: True*энтер*False*энтер*False*энтер*False, если квартира в пределах города) \n Все значения должны быть введены в булевых значениях типа True/False либо в целочисленных (в метрах и штуках)"


@bot.message_handler(commands=["start"])
def repeat_all_message(message):
    bot.reply_to(message, hellomessage)

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message):
    bot.send_message(message.chat.id, f"Стоимость квартиры составляет: {predict_price(message.text)[0].round(1)} рублей")



bot.polling(none_stop=True)