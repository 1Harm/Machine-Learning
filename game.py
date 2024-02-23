import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='adam',
              loss='mean_squared_error')

# Генерация случайного числа, которое игрок должен угадать
target_number = np.random.randint(1, 101)

def guess_number(player_guess):
    player_guess = np.array([player_guess])
    predicted_number = model.predict(player_guess)[0][0]

    if predicted_number < target_number:
        return "You won"
    elif predicted_number > target_number:
        return "Too high number"
    else:
        return "You won!"

# Начало игры
print("Welcome to the game predict the number'!")
print("Guess the number from 1 to 100.")

while True:
    try:
        player_input = int(input("Guess the number: "))
        result = guess_number(player_input)
        print(result)

        if result == "You won!":
            break
    except ValueError:
        print("Please enter valid number.")
