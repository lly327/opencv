import cv2
from util import plt_show, plt_show0, get_mean_luminance, adjust_img, tamper_plate, visual_approxPolyDP_cicle
import numpy as np
import imutils
import os
import math
import copy


def filtrate(gray, candidates, clearBorder=False):
    lpCnt = None  # 保存车牌轮廓
    roi = None  # 车牌感兴趣的区域
    minAR = 2
    maxAR = 4
    for c in candidates:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)  # 每个矩形的宽和高的比例
        if ar >= minAR and ar <= maxAR:
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            plt_show(roi)


def original(img_gray):
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, rectKern)  # 底帽运算，原图像减去闭运算结果
    # plt_show(blackhat)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, squareKern)  # 闭运算
    # plt_show(light)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # 检测轮廓，x方向梯度  CV_32F表示32位浮点数
    gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradX = gradX + gradY
    gradX = np.absolute(gradX)  # 取绝对值
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))  # 图片中的最小灰度和最大灰度
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))  # 当前像素点的值和最小灰度的差值 占 最大灰度差值的比例   再乘255
    gradX = gradX.astype('uint8')  # 转为8位无符号数
    # plt_show(gradX)

    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # plt_show(thresh)

    # 消除部分噪声,暗部细节部分消失
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # plt_show(thresh)

    # 突出画面中较亮的部分
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    # plt_show(thresh)

    return thresh


def blur(img):
    """几种常见的模糊方法"""
    # 均值滤波
    mean_blur = cv2.blur(img, (3, 3))

    # 高斯滤波
    # gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # 中值滤波
    # median_blur = cv2.medianBlur(img, 3)

    # 双边滤波
    # double_blur = cv2.bilateralFilter(img, 11, 17, 17)

    result = mean_blur
    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    return result


def edge_detection(img_gray):
    """比较几种不同的边缘检测算子"""
    # Sobel
    gradX = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # 检测轮廓，x方向梯度  CV_32F表示32位浮点数
    gradY = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)  # -1表示输出图像和原始图像一样大小
    gradX = gradX + gradY
    gradX = np.absolute(gradX)  # 取绝对值
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))  # 图片中的最小灰度和最大灰度
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))  # 当前像素点的值和最小灰度的差值 占 最大灰度差值的比例   再乘255
    gradX = gradX.astype('uint8')  # 转为8位无符号数
    # plt_show(gradX)

    # Scharr
    # scharr_x = cv2.Scharr(img_gray, cv2.CV_32F, 1, 0)
    # scharr_y = cv2.Scharr(img_gray, cv2.CV_32F, 0, 1)
    # scharr_x = cv2.convertScaleAbs(scharr_x)
    # scharr_y = cv2.convertScaleAbs(scharr_y)
    # scharr_whole = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
    # scharr_whole = scharr_x+scharr_y
    # plt_show(scharr_whole)

    # Laphlacian 拉普拉斯
    # laphlacian = cv2.Laplacian(img_gray, cv2.CV_32F)
    # laphlacian = cv2.convertScaleAbs(laphlacian)
    # plt_show(laphlacian)

    # Canny
    # canny_1 = cv2.Canny(img_gray, 64, 128)
    # canny_2 = cv2.Canny(img_gray, 100, 200)  # 大的阈值用于检测比较明显的边缘信息
    # plt_show(canny_1)
    # plt_show(canny_2)

    result = gradX
    # plt_show(result)

    return result


def process(img_color, img_gray):
    img_lumin = get_mean_luminance(img_color)
    if img_lumin >= 120:  # 室外车辆
        thresh_value = 90
    else:  # 室内车辆
        thresh_value = 70
    # 二值化
    # ret, binary = cv2.threshold(img_gray, 220, 0, cv2.THRESH_BINARY)
    ret, binary = cv2.threshold(img_gray, thresh_value, 255, cv2.THRESH_BINARY)  # 120
    # binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # plt_show(binary)
    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))  # 9 7
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=4)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=1)
    plt_show(dilation2)
    return dilation2


def second_filter_color(image):
    img_B = cv2.split(image)[0]
    img_G = cv2.split(image)[1]
    img_R = cv2.split(image)[2]
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = image.shape[0]
    w = image.shape[1]
    count = 0
    for i in range(h):
        for j in range(w):
            # 18, 161, 228
            if (img_B[i,j]/(img_G[i,j]+1) >= 1.3) and (img_B[i, j]>=80):
                count += 1
            # if ((img_HSV[:, :, 0][i, j] - 115) ** 2 < 15 ** 2) and (img_B[i, j] > 70) and (img_R[i, j] < 40):
            #     count += 1
            else:
                pass
    ratio = count / (h * w)
    print(ratio)
    return ratio


def findPlateNumberRegion(ori_img, img, plate_dir, img_name):
    region = []
    record_img_no_four = []
    flag = 1

    # 查找外部轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)  # 计算该轮廓的面积
        if area > 20000:  # 首先筛选掉面积小的
            min_rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (side1,side2), 旋转角度）
            approx = cv2.approxPolyDP(cnt, 110, True)  # 找轮廓的四个顶点

            side1 = int(min_rect[1][0])
            side2 = int(min_rect[1][1])
            height, width = (side1, side2) if side1 < side2 else (side2, side1)

            if (width > (height * 1.5)) and (width < (height * 6)) and len(approx) == 4:  # and len(approx)==4
                region.append([area, min_rect, approx])  # 只有被识别为四边形的轮廓 并且 其最小外接矩形的高度和宽度合适时，才会进入候选区

    if len(region) == 0:
        print(f"没有找到符合条件的轮廓:{img_name}")
        return None, 5

    if len(region) > 1:
        # 按照最小外接矩形的位置排序，位置越靠下，越有可能是车牌（减少车窗的干扰）, 排的越靠前
        region = sorted(region, key=lambda x: x[1][0][1], reverse=True)

    if len(region) > 0:
        min_rect = region[0][1]  # 轮廓的最小外接矩形（中心(x,y), (side1,side2), 旋转角度）
        approx = region[0][2]  # 轮廓的四个顶点
        # visual_approxPolyDP_cicle(ori_img, approx, 'cicle', flag=False)  # 可视化选到的四边形
        if len(approx) == 4:
            # 调用函数返回整体矫正后的图片
            adjusted_img, left_top_point, width, height, M, src_dst_points = adjust_img(ori_img, min_rect, approx, img_name)
            if adjusted_img is None:
                image = None
                print(f"adjusted_img is None  {adjusted_img}")
                return image, 6

            x = left_top_point[0]
            y = left_top_point[1]
            # plt_show0(adjusted_img)  # 整体图片经过透视变换后的样子

            image = adjusted_img[y:y + height, x:x + width]
            plt_show0(image)  # 车牌区域
            # image2 = copy.deepcopy(image)
            try:
                tamper_img = tamper_plate(image, img_name)  # 返回篡改后的车牌图片
                adjusted_img[y:y + height, x:x + width] = tamper_img
            except Exception as e:
                print(f"{e}篡改失败的图片：{img_name}")
                image = None
                return image, 5

            # plt_show0(tamper_img)  # 篡改后的车牌

            # plt_show0(adjusted_img)  # 将篡改后的车牌放回图片中

            try:
                dst = cv2.warpPerspective(adjusted_img, M, (ori_img.shape[1], ori_img.shape[0]),
                                          flags=cv2.WARP_INVERSE_MAP)
            except Exception as e:
                print(f'{e}将图片映射回原来大小错误:{img_name}')
                return None, 5
            # plt_show0(dst)

            flag = 1
            return dst, flag
            # plt_show0(image)
        else:
            flag = 2
            image = None
            print(f'approx!=4,当前mask中未发现四边形----{img_name}')
    else:
        flag = 3
        image = None
        print(f"未发现车牌信息，当前图片名：{img_name}")

    # 如果一张图片中有多个可能区域，那么筛选出面积最大的输出
    # if len(region) > 1:
    #     max_area = 0
    #     for box in region:
    #         if max_area < box[2]*box[3]:
    #             max_area = box[2]*box[3]
    #             # print(max_area)
    #             region[0] = box
    # if len(region) > 0:
    #     x = region[0][0]
    #     y = region[0][1]
    #     weight = region[0][2]
    #     height = region[0][3]
    #
    #     cv2.drawContours(ori_img, contours, region[0][4], (0, 255, 0), 5)
    #     plt_show0(ori_img)
    #
    #     image = ori_img[y:y + height, x:x + weight]
    #     plt_show0(image)
    # else:
    #     print(f"未发现车牌信息，当前图片名：{img_name}")

        # region.append(box)

    return image, flag


def draw_plate(img, region):
    # 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)

        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]

        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]

        img_org2 = img.copy()
        img_plate = img_org2[y1:y2, x1:x2]
        plt_show0(img_plate)

        # 带轮廓的图片
        plt_show0(img)


def main():
    img_gray = cv2.imread('/home/lily/EfficientNet_pytorch/platenumber/060.jpg', 0)  # 读取灰度图像
    img_gray = cv2.medianBlur(img_gray, 9)
    img_gray = cv2.blur(img_gray, (1, 15))
    img_gray = cv2.blur(img_gray, (15, 1))
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    close_img = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, rectKern)

    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, rectKern)  # 底帽运算，原图像减去闭运算结果

    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKern)  # 顶帽运算，原图像减去闭运算结果

    tidu = cv2.morphologyEx(tophat, cv2.MORPH_GRADIENT, rectKern, iterations=2)  # 底帽运算，原图像减去闭运算结果

    # plt_show(close_img)
    # plt_show(blackhat)
    # plt_show(tophat)
    # plt_show(tidu)

    # squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #
    # light = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, squareKern)

    light = cv2.threshold(tidu, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    result = cv2.dilate(light, rectKern, iterations=2)

    plt_show(result)


def traverse_img():
    img_dir = '/home/lily/EfficientNet_pytorch/platenumber/pic/'
    # img_names = os.listdir(img_dir)
    # print(img_names)
    img_names_outside = ['007.jpg', '054.jpg', '013.jpg', '059.jpg',
                 '004.jpg', '060.jpg', '079.jpg', '070.jpg',
                 '086.jpg', '095.jpg', '096.jpg', '106.jpg',
                 '107.jpg']
    img_names_inside = ['2021_06_23_12_43_IMG_5488.JPG', '2021_06_23_12_48_IMG_5516.JPG',
                        '2021_06_23_12_44_IMG_5493.JPG', '2021_06_23_12_43_IMG_5491.JPG',
                        '2021_06_23_12_44_IMG_5495.JPG', '2021_06_23_12_44_IMG_5494.JPG',
                        '2021_06_23_12_44_IMG_5496.JPG', '2021_06_23_12_45_IMG_5501.JPG',
                        '2021_06_23_12_44_IMG_5497.JPG', '2021_06_23_12_45_IMG_5499.JPG',
                        '2021_06_23_12_45_IMG_5498.JPG', '2021_06_23_12_45_IMG_5503.JPG',
                        '2021_06_23_12_45_IMG_5504.JPG', '2021_06_23_12_48_IMG_5518.JPG',
                        '2021_06_23_12_49_IMG_5519.JPG', '2021_06_23_12_51_IMG_5523.JPG']
    # img_names_inside = ['2021_06_23_12_44_IMG_5496.JPG', '2021_06_23_12_49_IMG_5519.JPG']
    img_names_outside = ['079.jpg']
    mean_lumin = []
    for img_name in img_names_outside:
        img_path = os.path.join(img_dir, img_name)
        img_color = cv2.imread(img_path)
        # img_gray = cv2.imread(img_path, 0)  # 读取灰度图像
        img_blur = blur(img_color)
        img_edge = edge_detection(img_blur)
        img = process(img_color, img_edge)
        region = findPlateNumberRegion(img_color, img)


def separate_blue():
    """直接定位图片中蓝色区域"""
    img_dir = '/data2/lily/datasets/car_plate_test/'
    img_names = os.listdir(img_dir)
    # img_names = sorted(img_names)
    np.random.shuffle(img_names)
    record_img_no_four = []
    record_img_no_ratio = []
    # print(img_names)
    img_names_outside = ['054.jpg', '059.jpg',
                 '060.jpg', '079.jpg', '070.jpg',
                 '086.jpg', '095.jpg', '096.jpg', '106.jpg',
                 '107.jpg']
    img_names_inside = ['2021_06_23_12_43_IMG_5488.JPG', '2021_06_23_12_48_IMG_5516.JPG',
                        '2021_06_23_12_44_IMG_5493.JPG', '2021_06_23_12_43_IMG_5491.JPG',
                        '2021_06_23_12_44_IMG_5495.JPG', '2021_06_23_12_44_IMG_5494.JPG',
                        '2021_06_23_12_44_IMG_5496.JPG', '2021_06_23_12_45_IMG_5501.JPG',
                        '2021_06_23_12_44_IMG_5497.JPG', '2021_06_23_12_45_IMG_5499.JPG',
                        '2021_06_23_12_45_IMG_5498.JPG', '2021_06_23_12_45_IMG_5503.JPG',
                        '2021_06_23_12_45_IMG_5504.JPG', '2021_06_23_12_48_IMG_5518.JPG',
                        '2021_06_23_12_49_IMG_5519.JPG', '2021_06_23_12_51_IMG_5523.JPG']
    # img_names_inside = ['2021_06_23_12_48_IMG_5518.JPG']
    # img_names_outside = ['086.jpg']
    # img_names = ['86.jpg', '93.jpg', '126.jpg', '133.jpg', '134.jpg', '135.jpg', '144.jpg']
    # img_names = ['03.jpg', '27.JPG', '43.jpg', '66.jpg', '75.jpg', '96.jpg', '114.jpg', '118.jpg']
    img_names = ['57.jpg']
    mean_lumin = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        # 用滤波
        # img = cv2.GaussianBlur(img, (5,5), 0)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lowerb = np.array([95, 145, 60])  # 50可以
        upperb = np.array([125, 255, 255])
        # lowerb = np.array([135, 145, 60])  # 50可以
        # upperb = np.array([165, 255, 255])
        mask = cv2.inRange(img_hsv, lowerb, upperb)
        # plt_show(mask)

        # 膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))  # (9 1)
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))  # (9 7) (13 5)
        # 膨胀一次，让轮廓突出
        dilation = cv2.dilate(mask, element2, iterations=3)
        # 腐蚀一次，去掉细节
        erosion = cv2.erode(dilation, element1, iterations=1)
        # 再次膨胀，让轮廓明显一些
        dilation2 = cv2.dilate(erosion, element2, iterations=1)

        # ship_masked = cv2.bitwise_and(img, img, mask=mask)
        # 转回RGB格式
        # ship_blue = cv2.cvtColor(ship_masked, cv2.COLOR_BGR2RGB)
        # plt_show(dilation2)
        mask_dir = '/data2/lily/datasets/car_plate_test_result/mask/'
        plate_dir = '/data2/lily/datasets/car_plate_test_result/plate/'
        tamper_dir = '/data2/lily/datasets/car_plate_test_result/tamper/'
        # cv2.imwrite(mask_dir+img_name, dilation2)
        # plt_show(dilation2)

        # plt_show(ship_masked)
        # plt_show(ship_blue)

        plate_image, flag = findPlateNumberRegion(img, dilation2, plate_dir, img_name)
        if flag == 1:
            try:
                print(f"success:{img_name}")
                cv2.imwrite(tamper_dir + img_name, plate_image)
            except Exception as e:
                print(f'{e}')
                print(img_name)
                continue
        elif flag == 2:
            record_img_no_four.append(img_name)
        else:
            record_img_no_ratio.append(img_name)

    print(f'在mask中没有发现四边形{len(record_img_no_four)}：{record_img_no_four}')
    print(f"在mask中没有发现合适比例的最小外接矩形{len(record_img_no_ratio)}: {record_img_no_ratio}")


if __name__ == '__main__':
    separate_blue()
    # traverse_img()
