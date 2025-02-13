import PIL.Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.metrics import mean_squared_error


# Funkcja do obliczenia MSE
def calculate_mse(original, reconstructed):
    # Obliczenie MSE tylko dla niezerowych wartości oryginalnego obrazu
    mask = original > 0
    return mean_squared_error(original[mask], reconstructed[mask])


# Modyfikacja głównej funkcji, aby dodać wybór metody interpolacji
def interpolate_and_evaluate(image, height, width, interpolation_function, filter_type):
    start_time = time.time()
    if filter_type == 'bayer':
        interpolated_image = interpolation_for_bayer(image.copy(), height, width, interpolation_function)
    elif filter_type == 'fuji':
        interpolated_image = nearest_neighbor_for_fuji(image.copy(), height, width)
    else:
        raise ValueError("Nieznany typ filtra")
    end_time = time.time()
    execution_time = end_time - start_time

    # Uwaga: 'cmos_data' powinno być obrazem bez masek, aby MSE było miarodajne
    mse = calculate_mse(cmos_data, interpolated_image)

    return interpolated_image, mse, execution_time


def apply_bayer_mask(image, height, width):
    bayer_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if i % 2 == 0 and j % 2 == 1:  # Piksele czerwone
                bayer_image[i, j, 0] = image[i, j, 2]  # R
            elif i % 2 == 1 and j % 2 == 0:  # Piksele niebieskie
                bayer_image[i, j, 2] = image[i, j, 0]  # B
            else:  # Piksele zielone
                bayer_image[i, j, 1] = image[i, j, 1]  # G
    return bayer_image


def apply_fuji_xtrans_mask(image, height, width):
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    xtrans_pattern = [
        [1, 0, 2, 1, 2, 0],
        [0, 1, 1, 2, 1, 1],
        [2, 1, 1, 0, 1, 1],
        [1, 2, 0, 1, 0, 2],
        [2, 1, 1, 0, 1, 1],
        [0, 1, 1, 2, 1, 1]
    ]
    for j in range(height):
        for i in range(width):
            index = xtrans_pattern[j % 6][i % 6]
            if index == 0:
                rgb_image[j, i, 0] = image[j, i, 2]
            elif index == 1:
                rgb_image[j, i, 1] = image[j, i, 1]
            else:
                rgb_image[j, i, 2] = image[j, i, 0]
    return rgb_image


def nearest_neighbor_for_bayer(rgb_image, i, j, channel):
    if channel == 0:  # dla czerwonego kanału
        if i != 0 and i % 2 == 1:
            nearest_i = i - 1
        else:
            nearest_i = i
        if j != 0 and j % 2 == 0:
            nearest_j = j - 1
        else:
            nearest_j = j
    elif channel == 2:  # dla niebieskiego
        if i != 0 and i % 2 == 0:
            nearest_i = i - 1
        else:
            nearest_i = i
        if j != 0 and j % 2 == 1:
            nearest_j = j - 1
        else:
            nearest_j = j
    else:
        if i != 0:
            if i % 2 == 1 and j % 2 == 0:
                nearest_i = i - 1
                nearest_j = j
            elif i % 2 == 0 and j % 2 == 1:
                nearest_i = i - 1
                nearest_j = j
        elif i == 0 and j % 2 == 1:
            nearest_i = i
            nearest_j = j - 1
        else:
            nearest_i = i
            nearest_j = j
    return rgb_image[nearest_i, nearest_j, channel]


def linear_interpolation_for_bayer(image, i, j, channel):
    height, width = image.shape[:2]
    total = 0
    count = 0
    if channel == 1:  # Dla zielonego
        if (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
            if 0 <= j - 1:
                total += image[i, j - 1, channel]
                count += 1
            if j + 1 < width:
                total += image[i, j + 1, channel]
                count += 1
    elif channel == 0:  # Dla czerwonego
        if i % 2 == 1 and j % 2 == 1:
            if 0 <= i - 1:
                total += image[i - 1, j, channel]
                count += 1
            if i + 1 < height:
                total += image[i + 1, j, channel]
                count += 1
        elif i % 2 == 0 and j % 2 == 0:
            if 0 <= j - 1:
                total += image[i, j - 1, channel]
                count += 1
            if j + 1 < width:
                total += image[i, j + 1, channel]
                count += 1
        else:
            if 0 <= i - 1 and 0 <= j - 1:
                total += image[i - 1, j - 1, channel]
                count += 1
            if i + 1 < height and j + 1 < width:
                total += image[i + 1, j + 1, channel]
                count += 1
    else:
        if i % 2 == 0 and j % 2 == 0:
            if 0 <= i - 1:
                total += image[i - 1, j, channel]
                count += 1
            if i + 1 < height:
                total += image[i + 1, j, channel]
                count += 1
        elif i % 2 == 1 and j % 2 == 1:
            if 0 <= j - 1:
                total += image[i, j - 1, channel]
                count += 1
            if j + 1 < width:
                total += image[i, j + 1, channel]
                count += 1
        else:
            if 0 <= i - 1 and 0 <= j - 1:
                total += image[i - 1, j - 1, channel]
                count += 1
            if i + 1 < height and j + 1 < width:
                total += image[i + 1, j + 1, channel]
                count += 1

    return total // count if count > 0 else 0


def interpolation_for_bayer(rgb_image, height, width, type_of_interpolation):
    for i in range(height):
        for j in range(width):
            if i % 2 == 0 and j % 2 == 1:  # Piksele czerwone
                rgb_image[i, j, 1] = type_of_interpolation(rgb_image, i, j, 1)  # Zielony
                rgb_image[i, j, 2] = type_of_interpolation(rgb_image, i, j, 2)
            elif i % 2 == 1 and j % 2 == 0:  # Piksele niebieskie
                rgb_image[i, j, 0] = type_of_interpolation(rgb_image, i, j, 0)  # Czerwony
                rgb_image[i, j, 1] = type_of_interpolation(rgb_image, i, j, 1)
            else:  # Piksele zielone
                rgb_image[i, j, 0] = type_of_interpolation(rgb_image, i, j, 0)  # Czerwony
                rgb_image[i, j, 2] = type_of_interpolation(rgb_image, i, j, 2)
    return rgb_image


def nearest_neighbor_for_fuji(rgb_image, height, width):
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            for channel in range(3):
                if rgb_image[y, x, channel] == 0:
                    # Get the neighboring pixel values in the same color channel
                    neighbors = rgb_image[y - 1:y + 2, x - 1:x + 2, channel]
                    neighbors = neighbors[neighbors > 0]
                    # Set the pixel value to the median of the neighbors if not empty, else 0
                    rgb_image[y, x, channel] = np.median(neighbors) if neighbors.size else 0

    return rgb_image


def quadratic_interpolation_for_bayer(rgb_image, x, y, channel):
    neighbors = []
    values = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            # Sprawdź, czy nie wyjdziemy poza obraz
            if 0 <= nx < rgb_image.shape[0] and 0 <= ny < rgb_image.shape[1]:
                neighbors.append((dx, dy))
                values.append(rgb_image[nx, ny, channel])

    # y = ax^2 + bxy + cy^2 + dx + ey + f
    A = []
    b = []
    for (dx, dy), v in zip(neighbors, values):
        A.append([dx * dx, dx * dy, dy * dy, dx, dy, 1])
        b.append(v)
    A = np.array(A)
    b = np.array(b)

    # system równań (parametry a, b, c, d, e, f)
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    interpolated_value = np.dot(coef, [0, 0, 0, 0, 0, 1])

    interpolated_value = np.clip(interpolated_value, 0, 255)

    return int(interpolated_value)


image_path = r"C:\Users\kfedo\Downloads\sioc3.jpg"
fig, ax = plt.subplots(1, 2, 3)

image = plt.imread(image_path)
cmos_data = image[:, :, [2, 1, 0]]
height, width = cmos_data.shape[:2]

obraz_bayer = apply_bayer_mask(cmos_data, height, width)
obraz_fuji = apply_fuji_xtrans_mask(cmos_data, height, width)

""" 
fig, ax = plt.subplots(1, 2) 
przyblizon_fuji =  
przyblizony_fuji = plt.imread(przyblizon_fuji) 

przyblizon_bayer =  
przyblizony_bayer = plt.imread(przyblizon_bayer) 

obraz_fuji = apply_fuji_xtrans_mask(cmos_data, height, width) 
ax[0].imshow(przyblizony_bayer) 
ax[0].set_title("Przybliżenie dla filtru kolorów Bayera") 
ax[1].imshow(przyblizony_fuji) 
ax[1].set_title("Przybliżenie dla filtru kolorów X-Trans") 
"""

interpolated_nn_bayer, mse_nn_bayer, time_nn_bayer = interpolate_and_evaluate(
    obraz_bayer, height, width, nearest_neighbor_for_bayer, 'bayer')
print("MSE for Bayer nearest neighbor:", mse_nn_bayer)
print("Execution time for Bayer nearest neighbor:", time_nn_bayer)

interpolated_linear_bayer, mse_linear_bayer, time_linear_bayer = interpolate_and_evaluate(
    obraz_bayer, height, width, linear_interpolation_for_bayer, 'bayer')
print("MSE for Bayer linear interpolation:", mse_linear_bayer)
print("Execution time for Bayer linear interpolation:", time_linear_bayer)

# Wywołanie funkcji i wyświetlenie wyników dla filtra Fuji
interpolated_nn_fuji, mse_nn_fuji, time_nn_fuji = interpolate_and_evaluate(
    obraz_fuji, height, width, None, 'fuji')  # 'interpolation_function' nie jest potrzebna dla Fuji
print("MSE for Fuji nearest neighbor:", mse_nn_fuji)
print("Execution time for Fuji nearest neighbor:", time_nn_fuji)

# interpolated_image1 = interpolation_for_bayer(obraz_bayer, height, width, nearest_neighbor_for_bayer)
# interpolated_image2 = interpolation_for_bayer(obraz_bayer, height, width, linear_interpolation_for_bayer)
interpolated_image3 = interpolation_for_bayer(obraz_bayer, height, width, quadratic_interpolation_for_bayer)

# interpolated_image4 = nearest_neighbor_for_fuji(obraz_fuji, height, width)

obra = r"C:\Users\kfedo\Pictures\Screenshots\Zrzut ekranu 2024-02-02 105149.png"
obraz = plt.imread(obra)
ax[0].imshow(interpolated_image3)
ax[0].set_title("Zinterpolowany obraz - metoda kwadratowa")
ax[1].imshow(obraz)
ax[1].set_title("Zinterpolowana kwadratowo macierz")

# dla interpolacki kwadratowej:
for i in range(height):
    for j in range(width):
        if i % 2 == 1 and j % 2 == 0:
            interpolated_image3[i, j, 0] = nearest_neighbor_for_bayer(interpolated_image3, i, j, 0)
        elif i % 2 == 0 and j % 2 == 1:
            interpolated_image3[i, j, 2] = nearest_neighbor_for_bayer(interpolated_image3, i, j, 2)

# średnia wartość dla każdego kanału
mean_red = np.mean(interpolated_image3[:, :, 0])
mean_green = np.mean(interpolated_image3[:, :, 1])
mean_blue = np.mean(interpolated_image3[:, :, 2])

# współczynniki do znormalizowania średnich wartości kanałów
normalize_factor = (mean_red + mean_blue) / 2
red_factor = normalize_factor / mean_red
green_factor = (normalize_factor / mean_green) * 0.97  # Lekko obniżamy zielony kanał
blue_factor = normalize_factor / mean_blue

# normalizacja
interpolated_image3[:, :, 0] = np.clip(interpolated_image3[:, :, 0] * red_factor, 0, 255)
interpolated_image3[:, :, 1] = np.clip(interpolated_image3[:, :, 1] * green_factor, 0, 255)
interpolated_image3[:, :, 2] = np.clip(interpolated_image3[:, :, 2] * blue_factor, 0, 255)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


gamma_corrected_rgb_image = adjust_gamma(interpolated_image3, gamma=1.3)

ax[2].imshow(interpolated_image3)
ax[2].set_title("Zinterpolowany obraz - metoda kwadratową + najbliższy sąsiad")

fig.tight_layout()

plt.show()