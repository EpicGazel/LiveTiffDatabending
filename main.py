# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import tifffile as tf
from primePy import primes as pr
from fibonacci import fibonacci
from pathlib import Path
import math
import numpy as np
import cv2
#from time import sleep, time
import time
import random

random.seed(time.time())


def alternate(image, rate):
    counter = rate
    colors = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]
    for y in range(image.shape[0]):
        color = colors[y % 3]
        for x in range(image.shape[1]):
            image[y][x] = color

            counter = rate_limiter(image, rate, counter)

    display_image(image)


def test_func(image):
    for i in range(25):
        for j in range(25):
            image[i][j] = np.array([255, 0, 0])
            display_image(image)


def flip(image, rate):
    counter = rate
    for y in range(image.shape[0]//2):  # y, 1080
        for x in range(image.shape[1]):  # x, 1920
            temp = np.copy(image[y, x])
            image[y][x] = image[image.shape[0] - y - 1][image.shape[1] - x - 1]
            image[image.shape[0] - y - 1][image.shape[1] - x - 1] = temp

            counter = rate_limiter(image, rate, counter)

    display_image(image)


def relative_brighten(image, rate):
    counter = rate
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            min_sum = np.sum(image[y][x])
            for i in range(3):
                for j in range(3):
                    if (y+i < image.shape[0]) and (x+j < image.shape[1]):
                        if np.sum(image[y+i][x+j]) < min_sum:
                            min_sum = 0
                            break
            if min_sum == np.sum(image[y][x]):
                total = 0
                avg = np.array([0.0, 0.0, 0.0])
                for i in range(3):
                    for j in range(3):
                        if (y + i < image.shape[0]) and (x + j < image.shape[1]):
                            if (not y == i) and (not x == j):
                                avg += image[y+i][x+j]
                                total += 1

                avg /= total
                np.flip(avg)
                #temp = "Image from " + str(image[y][x]) + " to "
                image[y][x] = avg
                #temp += str(image[y][x])
                #print(temp)

            #temp_pixel = np.copy(image[y][x])
            #image[y][x] = np.array([1.0, 0, 0])
            counter = rate_limiter(image, rate, counter)
            #image[y][x] = temp_pixel

    display_image(image)


def sine_effect(image, rate):
    counter = rate
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image[y][x] = image[y][x] \
                          * (np.array([math.sin(x), math.sin(y), math.sin(x+y)]) + 1)
            counter = rate_limiter(image, rate, counter)

    display_image(image)


def random_color_offsets(image, rate):
    counter = rate
    bOff, gOff, rOff = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
    print(str(bOff) + " " + str(gOff) + " " + str(rOff))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image[y][x] = image[y][x] \
                          + (np.array([bOff, gOff, rOff]) * 0.4)
            counter = rate_limiter(image, rate, counter)

    display_image(image)


def fib(image, rate):
    counter = rate
    fib_nums = fibonacci(image.shape[0] * image.shape[1])
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if y in fib_nums:
                image[y][x] = image[y][x] * np.array([0, 0, 2])
            elif x in fib_nums:
                image[y][x] = image[y][x] * np.array([0, 2, 0])
            else:
                image[y][x] = image[y][x]
            counter = rate_limiter(image, rate, counter)

    display_image(image)


def primes_filler(image, rate):
    counter = rate
    prime_pixel = image[0][2]
    offset = random.randint(-10, 10)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if pr.check(abs(x + offset)):
                prime_pixel = image[y][x]
            else:
                image[y][x] = prime_pixel

            counter = rate_limiter(image, rate, counter)

    display_image(image)


def primes(image, rate):
    counter = rate
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if pr.check(y):
                image[y][x] = image[y][x] * np.array([2.0, 2.0, 2.0])
            elif pr.check(x):
                image[y][x] = image[y][x] * np.array([1.5, 1.5, 1.5])

            counter = rate_limiter(image, rate, counter)

    display_image(image)


def resize_callback(event, x, y, flags, param):
    image = param
    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_MOUSEWHEEL:
        # Get the current size of the window
        width, height = cv2.getWindowImageRect('Test')[2:4]
        # Calculate the new height based on the aspect ratio
        #new_height = int((width / 16) * 9)
        new_height = int(width / (image.shape[1] / image.shape[0]))
        # Set the new size of the window
        cv2.resizeWindow('Test', width, new_height)


def rate_limiter(image, rate, counter):
    counter -= 1
    if counter <= 0:
        display_image(image)
        counter = rate

    return counter


def display_image(image):
    cv2.imshow('Test', image)
    cv2.waitKey(1)


def display_tiff(file_path):
    # Setup
    rate = 10000
    in_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    print(in_image.dtype)

    # avoid clamping
    image = in_image.astype(np.float64)  # Covert to float array
    image = image / 255.0  # Normalize

    cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Test', resize_callback, param=image)
    #cv2.resizeWindow('Test', 16*50, 9*50)
    #cv2.resizeWindow('Test', 9 * 50, 16 * 50)

    cv2.imshow('Test', image)
    cv2.waitKey(1)
    print(image.shape)

    # Main stuff
    edit_image(image, rate)

    # Clean up
    print('Done')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_prompt = input("Save?, 1 to save: ")
    if save_prompt == '1':
        output_file(image, rate)


def output_file(image, rate):
    counter = rate
    normalize = True
    if normalize:
        normalize_image(image, rate)
    else:  # Clamp
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for i in range (3):
                    image[y][x][i] = max(0.0, min(image[y][x][i], 1.0))

    #display_image(image)

    image *= 255.0
    image = image.astype(np.uint8)
    out_file_path = 'LiveTiffDatabending/data/out.png'
    path = Path(out_file_path)

    while path.is_file():
        out_file_path = out_file_path.replace(".", "1.")
        path = Path(out_file_path)

    cv2.imwrite(str(path), image)


def normalize_image(image, rate):
    counter = rate

    # Find Maxes
    maxs = image[0][0]
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x][0] > maxs[0]:
                maxs[0] = np.copy(image[y][x][0])
            if image[y][x][1] > maxs[1]:
                maxs[1] = np.copy(image[y][x][1])
            if image[y][x][2] > maxs[2]:
                maxs[2] = np.copy(image[y][x][2])

    # Normalize from maxs
    maxs = np.copy(maxs)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for i in range(3):
                image[y][x][i] = image[y][x][i] / maxs[i]
            counter = rate_limiter(image, rate, counter)


def edit_image(image, rate):
    # test_func(image)
    # alternate(image, rate)
    # flip(image, rate)
    # primes(image, rate)
    # fib(image, rate)
    # sine_effect(image, rate)
    # relative_brighten(image, rate)
    # flip(image, rate)
    # random_color_offsets(image, rate)
    for i in range(8):
        print("Primes: " + str(i))
        primes_filler(image, rate)
        # print("Normalize: " + str(i))
        # normalize_image(image, rate)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    display_tiff('LiveTiffDatabending/data/test3.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
