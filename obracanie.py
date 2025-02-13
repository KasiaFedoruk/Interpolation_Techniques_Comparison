import numpy as np
import matplotlib.pyplot as plt

# Wczytywanie obrazu
image_path = r"C:\Users\kfedo\Downloads\sioc3.jpg"
image = plt.imread(image_path)

# Pobieranie kąta obrotu i konwersja na radiany
kat = 36
kat_rad = np.deg2rad(kat)

# Tworzenie macierzy obrotu
macierz_obrotu = np.array([[np.cos(kat_rad), -np.sin(kat_rad)], [np.sin(kat_rad), np.cos(kat_rad)]])

# Znajdowanie środka obrazu
srodek = np.array([image.shape[1] / 2, image.shape[0] / 2])

def obrot_z_interpolacja(obraz, macierz_obrotu, srodek):
    obraz_obrocony = np.zeros_like(obraz)

    for x in range(obraz.shape[1]):
        for y in range(obraz.shape[0]):
            pozycja = np.array([x, y]) - srodek
            nowa_pozycja = np.dot(macierz_obrotu, pozycja) + srodek

            x1, y1 = int(nowa_pozycja[0]), int(nowa_pozycja[1])
            x2, y2 = x1 + 1, y1 + 1

            # Obliczanie wag dla czterech najbliższych pikseli
            waga1 = (x2 - nowa_pozycja[0]) * (y2 - nowa_pozycja[1])
            waga2 = (nowa_pozycja[0] - x1) * (y2 - nowa_pozycja[1])
            waga3 = (x2 - nowa_pozycja[0]) * (nowa_pozycja[1] - y1)
            waga4 = (nowa_pozycja[0] - x1) * (nowa_pozycja[1] - y1)

            # Obliczanie koloru nowego piksela
            if (x1 >= 0 and y1 >= 0 and x2 < obraz.shape[1] and y2 < obraz.shape[0]):
                obraz_obrocony[y, x] = waga1 * obraz[y1, x1] + waga2 * obraz[y1, x2] + waga3 * obraz[y2, x1] + waga4 * obraz[y2, x2]

    return obraz_obrocony

obraz_obrocony_raz = obrot_z_interpolacja(image, macierz_obrotu, srodek)

obraz_obrocony_10_razy = np.copy(image)

for i in range(10):
    obraz_obrocony_10_razy = obrot_z_interpolacja(obraz_obrocony_10_razy, macierz_obrotu, srodek)

roznica_obrazow = image - obraz_obrocony_10_razy

# Wyświetlanie obrazów
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Oryginalne zdjęcie')
ax[0].imshow(image)
ax[1].set_title('Obrócone zdjęcie')
ax[1].imshow(obraz_obrocony_raz)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].set_title('Obrócone zdjęcie 10 razy')
ax[0].imshow(obraz_obrocony_10_razy)
ax[1].set_title('Różnica obrazow')
ax[1].imshow(roznica_obrazow)
plt.tight_layout()
plt.show()