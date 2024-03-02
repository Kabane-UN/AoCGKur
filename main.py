import numpy
from PIL import Image
import numpy as np
import math


def gause_filter(img, height, width):
    res = np.zeros(shape=(height, width))
    Gause = np.array([[0.5, 0.75, 0.5], [0.75, 1.0, 0.75], [0.5, 0.75, 0.5]])
    for i in range(height - 2):
        for j in range(width - 2):
            res[i + 1][j + 1] = np.sum(np.multiply(Gause, img[i:i + 3, j:j + 3]) * 1 / 6)
    return res


def sobel_filter(img, height, width):
    res = np.zeros(shape=(height, width))
    angels = np.zeros(shape=(height, width))
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    for i in range(height - 2):
        for j in range(width - 2):
            gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))
            g = np.sqrt(gx ** 2 + gy ** 2)
            res[i + 1][j + 1] = g
            if g != 0:
                angels[i + 1][j + 1] = round(math.atan2(gy, gx) / (math.pi / 4)) * (math.pi / 4)
            else:
                angels[i + 1][j + 1] = float("NaN")
    return res, angels


def check(img, x, y, v):
    (height, width) = np.shape(img)
    if 0 <= x <= height - 1 and 0 <= y <= width - 1 and img[x][y] <= v:
        return True
    else:
        return False


def sign(val):
    if -0.000001 < val < -0.000001:
        return 0
    elif val < 0:
        return -1
    else:
        return 1


def NMS(img, angels, height, width):
    res = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            if math.isnan(angels[i][j]):
                continue
            dx = sign(math.cos(angels[i][j]))
            dy = sign(math.sin(angels[i][j]))
            if check(img, i + dx, j + dy, img[i][j]):
                res[i + dx][j + dy] = 0
            if check(img, i - dx, j - dy, img[i][j]):
                res[i - dx][j - dy] = 0
            res[i][j] = img[i][j]
    return res


def two_filtration(img, mi, ma):
    (height, width) = np.shape(img)
    step = np.zeros(shape=(height, width))
    res = np.zeros(shape=(height, width))
    for i in range(height):
        for j in range(width):
            p = img[i][j]
            if p <= mi:
                step[i][j] = 0
            elif p >= ma:
                step[i][j] = 255
            else:
                step[i][j] = p
    paths = [[-1, -1, -1, 0, 0, 1, 1, 1],
             [-1, 0, 1, -1, 1, -1, 0, 1]]
    for i in range(height):
        for j in range(width):
            p = step[i][j]
            if p == 255:
                res[i][j] = 255
                for k in range(8):
                    dx = paths[0][k]
                    dy = paths[1][k]
                    x = i
                    y = j
                    while True:
                        x += dx
                        y += dy
                        if x < 0 or x > height - 1 or y < 0 or x > width - 1 or step[x][y] == 0 or step[x][y] == 255:
                            break
                        res[x][y] = 255
            else:
                res[i][j] = 0
    return res


def find_ones(img):
    height, width = np.shape(img)
    res = []
    for j in range(width):
        for i in range(height):
            if img[i][j] > 0:
                res.append([j, i])
    return res


def delta(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5


def hough(img, min_b, max_b, min_votes):
    res = []
    height, width = np.shape(img)
    pixels = find_ones(img)
    acc = np.zeros(max(height, width) // 2)
    print(len(pixels))
    for p1 in range(len(pixels) - 1):
        for p2 in range(len(pixels) - 1, p1 + 2, -1):
            x1, y1 = pixels[p1][0], pixels[p1][1]
            x2, y2 = pixels[p2][0], pixels[p2][1]
            d = delta(x1, y1, x2, y2)
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            a = d / 2
            acc = acc * 0
            if x1 != x2 and max_b > d > min_b:
                alpha = math.atan((y2 - y1) / (x2 - x1))
                for p3 in range(len(pixels)):
                    if (p3 == p1) and (p3 == p2):
                        continue
                    x3, y3 = pixels[p3][0], pixels[p3][1]
                    d3 = delta(x3, y3, xc, yc)
                    if d3 >= a:
                        continue
                    f = delta(x3, y3, x2, y2)
                    if (2 * a * d3) == 0:
                        continue
                    cos_tau = ((a ** 2 + d3 ** 2 - f ** 2) / (2 * a * d3)) ** 2
                    sin_tau = 1 - cos_tau
                    bg = ((a ** 2 * d3 ** 2 * sin_tau) / (a ** 2 - d3 ** 2 * cos_tau)) ** .5
                    if type(bg) is complex:
                        bg = bg.real
                    b = round(bg)
                    if 0 < b < len(acc):
                        acc[b-1] += 1
                m = max(acc)
                if m > min_votes:
                    index, = np.where(acc == m)
                    index = index[0]
                    a = round(a)
                    B = index + 2
                    res.append([round(xc), round(yc), B, a, alpha])
                    if len(res) > 0:
                        return res
    return res


def draw_ellipse(el, img):
    stapn = 1000
    for step in range(1, stapn+1):
        o = 2 * np.pi * step / stapn
        x = el[2] * math.cos(o) + el[1]
        y = el[3] * math.sin(o) + el[0]
        fi = el[4]*np.pi/180
        X = x*math.cos(fi) - y*math.sin(fi)
        Y = y * math.cos(fi) + x * math.sin(fi)
        X = round(X)
        Y = round(Y)
        # X = round(x)
        # Y = round(y)
        try:
            img[X][Y] = 255
        except Exception as e:
            pass
    return img


path = r"photo/b.jpg"
img = np.asarray(Image.open(path))
height, width, _ = np.shape(img)
r_img, g_img, b_img = img[:, :, 0], img[:, :, 1], img[:, :, 2]
img = 0.299 * r_img + 0.587 * g_img + 0.114 * b_img
img, angels = sobel_filter(gause_filter(img, height, width), height, width)
img = NMS(img, angels, height, width)
img = two_filtration(img, 200, 250)
out_img = Image.fromarray(numpy.uint8(img))
out_img.show()
res = hough(img, 10, 50, 50)
plot = np.zeros((height, width), dtype=np.int32)
for r in res:
    plot = draw_ellipse(r, plot)
out_img = Image.fromarray(numpy.uint8(plot))
out_img.show()
