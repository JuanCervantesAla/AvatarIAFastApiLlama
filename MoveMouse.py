import pyautogui
import time

while True:
    # Obtiene la posición actual del ratón
    x, y = pyautogui.position()

    # Mueve el ratón 10 píxeles hacia la derecha y 10 píxeles hacia abajo
    pyautogui.moveTo(x + 10, y + 10)

    # Espera 10 segundos
    time.sleep(10)
