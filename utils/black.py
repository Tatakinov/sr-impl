#!/bin/python3

import cv2
import numpy as np
import random

size = 640

# 画像の作成（高さsizepx、幅sizepx、3チャンネルのカラー画像）
img = np.zeros((size, size, 3), dtype=np.uint8)
img[:] = (255, 255, 255) # 白い背景にする

# 作成する画像の数
num_images = 50

# 描画する図形の数
num_shapes = 10


for i in range(num_images):
    print('generate ' + str(i) + 'th image')
    shapes = []
    for _ in range(num_shapes):
        color = (0, 0, 0)

        # 図形の種類をランダムに選択 (0: 円, 1: 矩形, 2: 矢印, 3: 楕円, 4: ポリゴン)
        shape_type = random.randint(0, 4)
        line_type = [cv2.LINE_8, cv2.LINE_AA][random.randint(0, 1)]
        thickness = random.randint(1, 10)
        if shape_type == 2 or shape_type == 4:
            thickness = random.randint(1, 10)
        shapes.append((shape_type, line_type, thickness, color))
    shapes = sorted(shapes, key = lambda v: v[2])

    for shape_type, line_type, thickness, color in shapes:
        if shape_type == 0:
            center_x = random.randint(0, size)
            center_y = random.randint(0, size)
            radius = random.randint(10, 200)
            cv2.circle(img, (center_x, center_y), radius, color, thickness = thickness, lineType = line_type)
        elif shape_type == 1:
            x1 = random.randint(0, size)
            y1 = random.randint(0, size)
            x2 = random.randint(0, size)
            y2 = random.randint(0, size)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness = thickness, lineType = line_type)
        elif shape_type == 2:
            x1 = random.randint(0, size)
            y1 = random.randint(0, size)
            x2 = random.randint(0, size)
            y2 = random.randint(0, size)
            cv2.arrowedLine(img, (x1, y1), (x2, y2), color, thickness = thickness, line_type = line_type)
        elif shape_type == 3:
            x = random.randint(0, size)
            y = random.randint(0, size)
            hr = random.randint(10, 200)
            vr = random.randint(10, 200)
            angle = random.randint(0, 359)
            cv2.ellipse(img, ((x, y), (hr, vr), angle), color, thickness = thickness, lineType = line_type)
        elif shape_type == 4:
            points = []
            for _ in range(random.randint(3, 7)):
                x = random.randint(0, size)
                y = random.randint(0, size)
                points.append((x, y))
            cv2.polylines(img, [np.array(points)], True, color, thickness = thickness, lineType = line_type)

    # 画像の保存 (必要に応じて)
    cv2.imwrite('black' + str(i) + '.png', img)
