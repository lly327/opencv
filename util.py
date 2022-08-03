import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import scipy.ndimage as pyimg
import easyocr
# import sys
# sys.path.append('/home/lily/EfficientNet_pytorch/platenumber/TET_GAN/src/')
from TET_GAN.src.oneshotfinetune import main
from TET_GAN.src.test import te_main
from pygame import freetype
import pygame, pygame.locals
import sys
print(sys.path)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def plt_show0(img):
    # cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


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


def get_mean_luminance(img_color):
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_v = img_hsv[:, :, 2]
    whole_lumin = 0
    h = img_v.shape[0]
    w = img_v.shape[1]
    for i in range(h):
        for j in range(w):
            whole_lumin += img_v[i, j]
    mean_lumin = whole_lumin/(h*w)

    return mean_lumin


def GenCh1(f, val, width, height):
    img = Image.new("RGB", (width, height),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val,(0,0,0),font=f)  # 文字左上角，绘制字符，文字颜色，字体类型
    # img = img.convert("RGB")
    # img = img.resize((width, height))
    A = np.array(img)

    return A


def visual_approxPolyDP_cicle(ori_img, approx, style, flag=True):
    """

    Args:
        ori_img: image where the approx in
        approx: use approxPolyDP get points
        style: 'cicle' or 'line'
        flag: if False, the function will execute nothing

    Returns:

    """
    if flag:
        if style == 'cicle':
            approx = approx.reshape(approx.shape[0], 2)
            for peak in approx:
                peak = peak[0]
                cv2.circle(ori_img, tuple(peak), 10, (0, 0, 255), 5)
            plt_show0(ori_img)
        elif style == 'line':
            cv2.drawContours(ori_img, [approx], 0, (0, 255, 0), 5)
            plt_show0(ori_img)
    else:
        pass


def in_out_door_lumin():
    img_dir = '/home/lily/EfficientNet_pytorch/platenumber/pic/'
    # img_names = os.listdir(img_dir)
    # print(img_names)
    # img_names = ['007.jpg', '054.jpg', '013.jpg', '059.jpg',
    #              '004.jpg', '060.jpg', '079.jpg', '070.jpg',
    #              '086.jpg', '095.jpg', '096.jpg', '106.jpg',
    #              '107.jpg']
    img_names_inside = ['2021_06_23_12_43_IMG_5488.JPG', '2021_06_23_12_48_IMG_5516.JPG',
                        '2021_06_23_12_44_IMG_5493.JPG', '2021_06_23_12_43_IMG_5491.JPG',
                        '2021_06_23_12_44_IMG_5495.JPG', '2021_06_23_12_44_IMG_5494.JPG',
                        '2021_06_23_12_44_IMG_5496.JPG', '2021_06_23_12_45_IMG_5501.JPG',
                        '2021_06_23_12_44_IMG_5497.JPG', '2021_06_23_12_45_IMG_5499.JPG',
                        '2021_06_23_12_45_IMG_5498.JPG', '2021_06_23_12_45_IMG_5503.JPG',
                        '2021_06_23_12_45_IMG_5504.JPG', '2021_06_23_12_48_IMG_5518.JPG',
                        '2021_06_23_12_49_IMG_5519.JPG', '2021_06_23_12_51_IMG_5523.JPG']
    mean_lumin = []
    for img_name in img_names_inside:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img_lumin = get_mean_luminance(img)
        mean_lumin.append(img_lumin)
    print(mean_lumin)
    print(np.mean(mean_lumin))

    # 室外的平均亮度 139.57324251697452
    # [148.5466774643314, 140.5543314870836, 160.8582968652099, 132.19416461074562,
    # 160.55879067378734, 127.28709061064119, 131.02979780179615, 141.94231458773888,
    # 124.4892511837863, 129.81808300862835, 131.33239307578842, 139.5745037846896, 146.26645756644214]

    # 室内的平均亮度 97.8784852371094
    # [107.09162972673637, 82.73411361554653, 94.9117270992116, 100.35177885776224,
    # 97.57355237137293, 92.41666535441337, 88.13460126527463, 101.03323995010813,
    # 81.95368639836336, 96.86574295516817, 105.21768707482993, 108.74600541894999,
    # 116.52114392728542, 83.94923654743533, 109.99047517347988, 98.56447805781264]


def separate_blue():
    img_path = '/home/lily/EfficientNet_pytorch/platenumber/pic/054.jpg'
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerb = np.array([100,100,100])
    upperb = np.array([140,255,255])
    mask = cv2.inRange(img_hsv, lowerb, upperb)
    ship_masked = cv2.bitwise_and(img, img, mask=mask)
    # 转回RGB格式
    ship_blue = cv2.cvtColor(ship_masked, cv2.COLOR_BGR2RGB)
    plt_show(mask)
    # plt_show(ship_masked)
    # plt_show(ship_blue)


def resort_point(arr, index):
    arr_len = len(arr)
    new_point = []
    for i in range(index, index + arr_len):
        if i <= arr_len-1:
            new_point.append(arr[i])
        else:
            new_point.append(arr[i - arr_len])
    return new_point


def lly_warpAffine(M, point):
    M11, M12, M13, M21, M22, M23 = M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2]
    x, y = point[0], point[1]
    newx = int(x*M11+y*M12+M13)
    newy = int(x*M21+y*M22+M23)

    return [newx, newy]


def lly_warpPerspective(M, point, delta_x, delta_y):
    # M是一个3*3的变换矩阵, point是一个点的坐标, delta是偏移值
    point.append(1.0)
    point = np.array(point).astype(np.float32)  # 1*3矩阵
    point = point[:, np.newaxis]
    u = M[0].dot(point)  # 1*3 3*1 = 1
    v = M[1].dot(point)  # 1*3 3*1 = 1
    w = M[2].dot(point)  # 1*3 3*1 = 1
    assert delta_x >= 0, print('x轴偏移量应该大于等于0')
    assert delta_y >= 0, print('y轴偏移量应该大于等于0')
    x = int(u/w + delta_x)  # 经过透视函数和偏移量之后的横坐标
    y = int(v/w + delta_y)  # 纵坐标

    return [x, y]


def change_perspective_M(M, delta_x, delta_y):
    """把透视函数的矩阵修改为加上偏移量的矩阵"""
    M[0] = M[0]+delta_x*M[2]
    M[1] = M[1]+delta_y*M[2]

    return M


def adjust_img(ori_img, min_rect, approx, img_name):
    delta = 7  # 定位车牌前由于加上了膨胀，导致车牌可能偏大，这个值用来减少这种偏差
    # 获取轮廓四个顶点坐标
    approx = approx.reshape(4, 2)
    approx_list = approx.tolist()

    side1 = int(min_rect[1][0])
    side2 = int(min_rect[1][1])
    rotate_angle = round(min_rect[2], 2)  #最小外接矩形的旋转角度
    angle_threshold = 85.0  # 对于这个临界值两侧的情况分开讨论
    # 先把点按照x+y由小到大的顺序排序,左上角的点总是在第一个位置，其次是左下，右上，右下
    approx_list = sorted(approx_list, key=lambda x: sum(x))
    # 1.旋转角度较小
    if rotate_angle >= angle_threshold:
        # 左上，左下，右下，右上
        point1, point2, point3, point4 = approx_list[0], approx_list[1], approx_list[3], approx_list[2]
        point1 = [point1[0]+delta, point1[1]+delta]  # 尽量精确定位车牌，减少膨胀影响
        point2 = [point2[0]+delta, point2[1]-delta]
        point3 = [point3[0]-delta, point3[1]-delta]
        point4 = [point4[0]-delta, point4[1]+delta]

        # 调整后的矩形长宽，应该看p1和p4两点之间的距离，而不是最小外接矩形的长宽
        temp_dis = np.array(point1)-np.array(point2)
        temp_dis = np.linalg.norm(temp_dis)  # 计算p1和p2之间的距离,就是高度
        temp_dis = int(temp_dis)
        height = temp_dis
        width = int(height*3.14)
    # 2.正倾斜 的矩形或者平行四边形
    elif side1 < side2 and rotate_angle < angle_threshold:
        point1, point2, point3, point4 = approx_list[0], approx_list[1], approx_list[3], approx_list[2]
        point1 = [point1[0] + delta, point1[1] + delta]  # 尽量精确定位车牌，减少膨胀影响
        point2 = [point2[0] + delta, point2[1] - delta]
        point3 = [point3[0] - delta, point3[1] - delta]
        point4 = [point4[0] - delta, point4[1] + delta]
        temp_dis = np.array(point1) - np.array(point4)
        temp_dis = np.linalg.norm(temp_dis)  # 计算p1和p4之间的距离
        temp_dis = int(temp_dis)
        width = temp_dis
        height = int(width/3.14)
    # 负倾斜
    elif side1 >= side2 and rotate_angle < angle_threshold:
        # approx_list = resort_point(approx_list, min_xy_index)
        point1, point2, point3, point4 = approx_list[0], approx_list[1], approx_list[3], approx_list[2]
        point1 = [point1[0] + delta, point1[1] + delta]  # 尽量精确定位车牌，减少膨胀影响
        point2 = [point2[0] + delta, point2[1] - delta]
        point3 = [point3[0] - delta, point3[1] - delta]
        point4 = [point4[0] - delta, point4[1] + delta]
        temp_dis = np.array(point1) - np.array(point4)
        temp_dis = np.linalg.norm(temp_dis)  # 计算p1和p4之间的距离
        temp_dis = int(temp_dis)
        width = temp_dis
        height = int(width / 3.14)
    else:
        print("车牌垂直！！！")
        # min_xy_index = approx_list_sum[1][1]
        # approx_list = resort_point(approx_list, min_xy_index)
        # p1为左上角坐标
        point1, point2, point3, point4 = approx_list[0], approx_list[1], approx_list[3], approx_list[2]
        width = side1
        height = side2

    new_point1 = point1
    new_point2 = [new_point1[0], new_point1[1]+height]
    new_point3 = [new_point2[0]+width, new_point2[1]]
    new_point4 = [new_point3[0], new_point3[1]-height]

    # 使用透视函数，得到透视变换的矩阵M
    src_four_points = np.float32([point1, point2, point3, point4])
    dst_four_points = np.float32([new_point1, new_point2, new_point3, new_point4])
    M = cv2.getPerspectiveTransform(src_four_points, dst_four_points)

    # 计算仿射变换后的图像的高度和宽度
    peak1, peak2, peak3, peak4 = [0, 0], [0, ori_img.shape[0] - 1], \
                                 [ori_img.shape[1] - 1, ori_img.shape[0] - 1], [ori_img.shape[1] - 1, 0]
    # peak1, peak2, peak3, peak4 = np.array(peak1).astype(np.float32), np.array(peak2).astype(np.float32), \
    #                              np.array(peak3).astype(np.float32), np.array(peak4).astype(np.float32),
    new_peak1, new_peak2, new_peak3, new_peak4 = lly_warpPerspective(M, peak1, 0, 0), \
                                                 lly_warpPerspective(M, peak2, 0, 0), \
                                                 lly_warpPerspective(M, peak3, 0, 0), \
                                                 lly_warpPerspective(M, peak4, 0, 0)
    new_peak = [new_peak1, new_peak2, new_peak3, new_peak4]

    # 计算变换后的点在x和y轴负方向上的最远距离
    min_x = sorted(new_peak, key=lambda x: x[0])[0][0]  # 首元素是四个点中横坐标最小的点
    min_y = sorted(new_peak, key=lambda x: x[1])[0][1]  # 首元素是四个点中纵坐标最小的点
    min_x = abs(min_x) if min_x<0 else 0
    min_y = abs(min_y) if min_y<0 else 0
    M = change_perspective_M(M, min_x, min_y)  # 修改透视函数M矩阵

    # 透视变换后图片的最大高度和宽度
    temp_shape_width = max(abs(new_peak1[0]-new_peak3[0]), abs(new_peak1[0]-new_peak4[0]),
                           abs(new_peak2[0]-new_peak3[0]), abs(new_peak2[0]-new_peak4[0]),
                           new_peak3[0], new_peak4[0])
    temp_shape_height = max(abs(new_peak1[1]-new_peak2[1]), abs(new_peak1[1]-new_peak3[1]),
                            abs(new_peak4[1]-new_peak2[1]), abs(new_peak4[1]-new_peak3[1]),
                            new_peak2[1], new_peak3[1])

    dst = cv2.warpPerspective(ori_img, M, (temp_shape_width, temp_shape_height))
    # plt_show0(dst)

    point1 = lly_warpPerspective(M, point1, 0, 0)  # 车牌左上角经过M矩阵(修改后的)后的坐标

    return dst, point1, width, height, M, [src_four_points, dst_four_points]


def get_tamper_char_mean_color(img, reverse=False):
    """提取需要篡改的字符的平均颜色"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if reverse:  # 如果要获取车牌背景颜色，要将轮廓扩大一些
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # (9 1)
        gray = cv2.dilate(gray, element1, iterations=1)

    ret2, binary2 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy.reshape((hierarchy.shape[1], hierarchy.shape[2]))
    approx_list = []
    for i in range(len(contours)):
        cnt = contours[i]
        if hierarchy[i][3] == -1:  # 由于在遍历父轮廓的时候对子轮廓已经处理了，所以不处理子轮廓
            approx = cv2.approxPolyDP(cnt, 3, False)  # 计算当前父轮廓的点集，每3个像素画一个点
            approx = approx.reshape(approx.shape[0], 2)
            area = cv2.contourArea(cnt)  # 计算当前轮廓的面积
            if hierarchy[i][2] != -1:  # 如果当前轮廓的子轮廓存在
                child_cnt = contours[hierarchy[i][2]]  # 子轮廓的cnt
                approx_child = cv2.approxPolyDP(child_cnt, 3, False)  # 用点逼近轮廓形状
                approx_child = approx_child.reshape(approx_child.shape[0], 2)
                approx = np.vstack((approx, approx_child))  # 将父子轮廓合并
            approx_list.append([approx, area])  # 将符合条件的轮廓信息添加到一个列表中
    approx_list = sorted(approx_list, key=lambda x: x[1], reverse=True)
    mask_img = np.zeros_like(gray)
    cv2.fillPoly(mask_img, [approx_list[0][0]], 255)  # 生成要篡改字符的mask
    if reverse:
        # mask_img = cv2.bitwise_not(mask_img)
        for i in range(mask_img.shape[0]):
            for j in range(mask_img.shape[1]):
                if mask_img[i, j] == 0:
                    mask_img[i, j] = 255
                else:
                    mask_img[i, j] = 0
    # plt_show(mask_img)
    mean_color = cv2.mean(img, mask_img)  # mask的字符范围比真实图像大，所以得到的平均字符颜色不够精确,bgr
    # print(mean_color)

    return mean_color


class draw_text_in_surf():
    def __init__(self, font='', surf_size=(320, 320), font_size=30):
        freetype.init()
        self.font = freetype.Font(font)
        self.font.antialiased = True
        self.font.origin = True
        self.font.size = font_size
        self.surf_size = surf_size

    def draw_plate_mask(self, text='A'):
        space = self.font.get_rect('O')
        char_box = self.font.get_rect(text)  # 获取单个字符的高度
        delta_char_height = self.surf_size[1] - char_box.height  # 画布高度和当前字符高度差
        y = int(delta_char_height / 2) + space.y  # 当前字符的y值，为了居中显示
        surf = pygame.Surface(self.surf_size, pygame.locals.SRCALPHA, 32)  # ((宽，高),显示格式，色深)
        if text == '1':
            x = 155
        else:
            x = 110
        self.font.render_to(surf, (x, y), text)
        surf_arr = pygame.surfarray.pixels_alpha(surf)  # 将字体设置为黑底白字 0黑255白
        surf_arr = surf_arr.swapaxes(0, 1)  # 翻转过来

        return surf_arr



def bw2dis(bw_img):
    """将黑白mask转为距离图"""
    surf_arr = np.expand_dims(bw_img, -1)
    surf_arr = surf_arr.repeat(3, axis=2)
    BW = surf_arr[:, :, 0] > 127  # 二值化图
    G_channel = pyimg.distance_transform_edt(BW)
    G_channel[G_channel > 255] = 255
    B_channel = pyimg.distance_transform_edt(1 - BW)
    B_channel[B_channel > 255] = 255
    surf_arr[:, :, 1] = G_channel.astype('uint8')
    surf_arr[:, :, 2] = B_channel.astype('uint8')
    r, g, b = surf_arr[:, :, 0], surf_arr[:, :, 1], surf_arr[:, :, 2]
    # img_new = cv2.merge([r, g, b])
    img_cv = cv2.merge([b, g, r])

    return img_cv


def use_tet_gan(img, bg_cor_bgr, char):
    """
    车牌需要篡改的字符图像，背景颜色rgb
    1. 将图像大小调整为320*320
    2. 生成要篡改的字符距离mask图 320*320
    3. 将两张图片拼接起来 左2右1
    4. tet-gan 的 oneshot 训练20epochs
    5. tet-gan 的 predict 输出256*256, resize为原始大小
    """
    plt_show0(img)
    ori_h = img.shape[0]
    ori_w = img.shape[1]
    # 1
    bg_img = np.zeros((320, 320, 3)).astype(np.uint8)
    for i in range(bg_img.shape[0]):
        for j in range(bg_img.shape[1]):
            for k in range(3):
                bg_img[i, j, k] = bg_cor_bgr[k]
    plt_show0(bg_img)

    ratio = 280/ori_h
    new_img = cv2.resize(img, (int(ratio * ori_w), 280))
    x = int((320-new_img.shape[1])/2)
    y = int((320-new_img.shape[0])/2)
    bg_img[y:y+new_img.shape[0], x:x+new_img.shape[1]] = new_img
    plt_show0(bg_img)  # resize 后 (高280,宽按比例)
    # 创建reader对象
    reader = easyocr.Reader(['ch_sim', 'en'])
    # 读取图像
    result = reader.readtext(bg_img)
    # 结果
    ori_text = result[0][1]

    # 2
    font = '/home/lily/SRNet-datagen/Synthtext/data/fonts/plate_ttf/QICHECHEPAIZT-B02S.ttf'
    dtis = draw_text_in_surf(font, (320, 320), 330)  # 字体类型，画布大小，字体大小
    surf_arr = dtis.draw_plate_mask(ori_text)
    plt_show(surf_arr)  # (320,320)

    img_cv = bw2dis(surf_arr)  # 将单通道的黑白图片转为距离图
    plt_show0(img_cv)

    # 3 拼接 img_cv bg_img
    whole_img = np.zeros((320, 640, 3)).astype(np.uint8)
    whole_img[0:320, 0:320, :] = img_cv
    whole_img[0:320, 320:, :] = bg_img
    plt_show0(whole_img)

    # 4 oneshot
    filename = '/home/lily/EfficientNet_pytorch/platenumber/TET_GAN/data/oneshotstyle/00.png'
    cv2.imwrite(filename, whole_img)
    main(filename)  # finetune

    # 5 predict
    content_mask = dtis.draw_plate_mask(char)
    content_mask = bw2dis(content_mask)  # 距离图
    filename_style = '/home/lily/EfficientNet_pytorch/platenumber/TET_GAN/data/style/00.png'
    filename_content = '/home/lily/EfficientNet_pytorch/platenumber/TET_GAN/data/content/00.png'
    cv2.imwrite(filename_style, bg_img)
    cv2.imwrite(filename_content, content_mask)
    style_name = filename_style  # 篡改原图 320 320
    content_name = filename_content  # 字符2 320 320
    content_type = 0  # 字符2类型，0距离图，1黑白图
    result_dir = '/home/lily/EfficientNet_pytorch/platenumber/TET_GAN/output/'
    name = '00.png'  # 预测结果名称
    te_main(style_name=style_name, content_name=content_name, content_type=content_type, result_dir=result_dir, name=name)

    result_path = os.path.join(result_dir, name)  # 256 256
    result_img = cv2.imread(result_path)
    plt_show0(result_img)
    result_img = result_img[:, 50:200, ]
    plt_show0(result_img)
    result_img = cv2.resize(result_img, (ori_w, ori_h))  # 将图片裁剪为原始大小
    plt_show0(result_img)

    print("INSIDE DONE")

    return result_img


def tamper_plate(ori_img, image_name):
    PLATE_CHARS_LETTER = ["A", "B", "C", "D", "E", "F", "G",
                          "H", "J", "K", "L", "M", "N",
                          "P", "Q", "R", "S", "T",
                          "U", "V", "W", "X", "Y", "Z"]
    PLATE_CHARS_NUMS_LETTER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                               "A", "B", "C", "D", "E", "F", "G",
                               "H", "J", "K", "L", "M", "N",
                               "P", "Q", "R", "S", "T",
                               "U", "V", "W", "X", "Y", "Z"]
    font_ch = '/home/lily/EfficientNet_pytorch/gen-Chinese-plate/font/platechar.ttf'
    font_en = '/home/lily/EfficientNet_pytorch/gen-Chinese-plate/font/platech.ttf'
    fontC = ImageFont.truetype(font_ch, 43, 0)
    fontE = ImageFont.truetype(font_en, 120, 0)
    # img_path = '/data2/lily/datasets/car_plate_test_result/plate/00.JPG'
    # ori_img = cv2.imread(img_path)
    approx_list = []  # 存储所有字符轮廓
    # plt_show0(ori_img)
    count = 0
    licenses = []
    gray2 = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros_like(gray2)
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # plt_show(gray2)

    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # (9 1)
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))  # (9 7) (13 5)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(gray2, element1, iterations=1)
    # erosion = cv2.erode(erosion, element2, iterations=1)
    ret2, binary2 = cv2.threshold(erosion, 120, 255, cv2.THRESH_BINARY)
    # plt_show(binary2)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(binary2, element1, iterations=1)
    dilation2 = cv2.dilate(dilation2, element2, iterations=1)
    # plt_show(dilation2)

    ret2, binary2 = cv2.threshold(dilation2, 120, 255, cv2.THRESH_BINARY)
    plt_show(binary2)  # 展示二值化后的图片

    contours, hierarchy = cv2.findContours(binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # RETR_CCOMP模式下只显示两级关系
    # contours, hierarchy = cv2.findContours(binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # RETR_LIST所有关系
    hierarchy = hierarchy.reshape((hierarchy.shape[1], hierarchy.shape[2]))
    for i in range(len(contours)):
        cnt = contours[i]
        if hierarchy[i][3] == -1:  # 由于在遍历父轮廓的时候对子轮廓已经处理了，所以不处理子轮廓
            area = cv2.contourArea(cnt)  # 计算当前轮廓的面积
            if area > 600:  # 只处理面积大于600的轮廓
                approx = cv2.approxPolyDP(cnt, 3, False)  # 计算当前父轮廓的点集，每3个像素画一个点
                min_rect = cv2.minAreaRect(cnt)  # 计算轮廓的最小外接矩形 （中心(x,y), (side1,side2), 旋转角度）
                points = cv2.boxPoints(min_rect)
                points = np.int0(points)  # 最小外接矩形的四个点坐标
                x = min_rect[0][0]  # 最小外接矩形的中心点x值

                side1 = int(min_rect[1][0])
                side2 = int(min_rect[1][1])
                height, width = (side1, side2) if side1 < side2 else (side2, side1)
                ratio = width / height + 0.001  # 轮廓的长宽之比，加0.001是为了防止除零报错

                if approx.shape[0] >= 3 and ratio > 1.6 and ratio < 2.3 and height > 20:  # 符合条件的才会被筛选进来
                    count += 1  # 记录有多少个符合条件，期望值为6
                    approx = approx.reshape(approx.shape[0], 2)

                    if hierarchy[i][2] != -1:  # 如果当前轮廓的子轮廓存在
                        child_cnt = contours[hierarchy[i][2]]  # 子轮廓的cnt
                        approx_child = cv2.approxPolyDP(child_cnt, 3, False)  # 用点逼近轮廓形状
                        approx_child = approx_child.reshape(approx_child.shape[0], 2)
                        approx = np.vstack((approx, approx_child))  # 将父子轮廓合并

                    approx_list.append([approx, x, min_rect, height, width, points])  # 将符合条件的轮廓信息添加到一个列表中
                    # cv2.fillPoly(ori_img, [approx], (0,0,255))  # 展示填充轮廓后的效果
                    # cv2.drawContours(ori_img, cnt, -1, (0, 255, 255), 2)  # 以线段方式展示选择的轮廓

                    # plt_show0(ori_img)
                    # print('hello')

    # plt_show0(ori_img)
    # print(f"=======count======: {count}")
    approx_list = sorted(approx_list, key=lambda x: x[1])  # 对字符轮廓进行排序，按照横坐标大小，小的在前，大的在后
    # assert len(approx_list) != 0, print(f"找不到字符轮廓{image_name}")
    if len(approx_list) != 0:
        index = np.random.randint(0, len(approx_list), 1)[0]
    else:
        dst_img = None
        print(f"找不到字符轮廓{image_name}")
        return dst_img

    if index == 0:  # 只能在字母中找
        index_first = np.random.randint(0, 24, 1)[0]
        char = PLATE_CHARS_LETTER[index_first]
    else:  # 在字母和数字中找 除去O和I
        index_first = np.random.randint(0, 32, 1)[0]
        char = PLATE_CHARS_NUMS_LETTER[index_first]
    # print(f"======char======{char}")

    # approx = approx_list[index][0]  # 选择需要篡改字符进行inpainting
    approx = approx_list[index][5]  # 选择需要篡改字符的最小外接矩形进行inpainting
    # TODO：可以对字符轮廓再进行扩充，让后面的inpainting可以得到更好的效果

    # 将篡改部分字符的平均颜色提取出来,输入参数:字符区域图像
    y = approx_list[index][2][0][1]  # 最小外接矩形的中心点y值
    x = approx_list[index][2][0][0]  # 最小外接矩形的中心点x值
    width = approx_list[index][4]
    height = approx_list[index][3]
    y = int(y-width/2)
    x = int(x-height/2)
    char_img = ori_img[y:y+width, x:x+height, :]
    plt_show0(char_img)  # 展示需要篡改的字符位置
    mean_color = get_tamper_char_mean_color(char_img)
    mean_bg_color = get_tamper_char_mean_color(char_img, reverse=True)  # reverse=True表示取蓝色背景的平均颜色

    # 这里对需要篡改的字符进行TET-GAN修改操作
    # tet_gan_char_img = use_tet_gan(char_img, mean_bg_color, char.upper())
    # ori_img[y:y + width, x:x + height, :] = tet_gan_char_img
    # dst_img = ori_img

    # 对需要篡改的区域进行inpainting操作
    cv2.fillPoly(mask_img, [approx], 255)  # 生成要篡改字符的mask
    # dst_img = cv2.inpaint(ori_img, mask_img, 3, cv2.INPAINT_TELEA)  # 这个inpainting方法效果更好
    for i in range(char_img.shape[0]):
        for j in range(char_img.shape[1]):
            for k in range(3):
                if mask_img[y + i, x + j] == 255:
                    ori_img[y + i, x + j, k] = mean_bg_color[k]

    # plt_show0(dst_img)

    min_rect = approx_list[index][2]
    points = cv2.boxPoints(min_rect)
    points = np.int0(points)  # 最小外接矩形的四个点坐标
    points = sorted(points, key=lambda x: x[0] + x[1])
    point1 = points[0]  # 最小外接矩形的左上角坐标
    x, y = point1[0], point1[1]
    width, height = approx_list[index][3], approx_list[index][4]  # 注意：这里把长的当做高，短边当做宽

    # 选择需要篡改的png
    # char = 'x'
    # print(char.upper())
    tamper_char = cv2.imread(f'/home/lily/EfficientNet_pytorch/platenumber/LicensePlateChars/AlpChars/{char.upper()}.png')
    # plt_show0(tamper_char)
    tamper_char = cv2.resize(tamper_char, (width, height))
    tamper_char = cv2.blur(tamper_char, (3, 3))  # 对篡改的字符进行平滑处理
    # plt_show0(tamper_char)
    for i in range(height):
        for j in range(width):
            for k in range(3):
                if tamper_char[i, j, k] == 0:
                    # temp_int = np.random.randint(-5, 6, 1)[0]
                    dst_img[y + i, x+5 + j, k] = mean_color[k]

    # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGBA)
    # temp = GenCh1(fontE, char, width, height)
    # temp_bgr = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
    # plt_show0(temp_bgr)
    # temp_bgr = cv2.resize(temp_bgr, (width, height))
    # plt_show0(temp_bgr)
    # dst_img[y:y + height, x:x + width, :] = temp
    # dst_img[x:x + width, y:y + height] = temp
    # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGBA2BGR)
    # for i in range(height):
    #     for j in range(width):
    #         for k in range(3):
    #             if temp[i, j, k] == 0:
    #                 dst_img[y + i, x + j, k] = 255

    # dst_img2 = cv2.inpaint(ori_img, mask_img, 3, cv2.INPAINT_NS)
    # plt_show(mask_img)
    # plt_show0(dst_img)
    # print(count)
    # print("DONE")

    return dst_img


def test_min_area_rect():
    test_dir = '/home/lily/EfficientNet_pytorch/platenumber/test2'
    imgs_name = os.listdir(test_dir)
    imgs_name = sorted(imgs_name)
    # imgs_name = ['9正1矩形.jpg']

    for img_name in imgs_name:
        print()
        docCnt = None
        img_path = os.path.join(test_dir, img_name)
        ori_img = cv2.imread(img_path)
        # 灰度方式读取
        img = cv2.imread(img_path, 0)
        # 二值化
        ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        img = cv2.imread('/home/lily/EfficientNet_pytorch/platenumber/output/outdoor/mask/086.jpg', 0)
        plt_show(img)
        # 查找轮廓
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 找到最小外接矩形
        for i in range(len(contours)):
            cnt = contours[i]
            min_rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            points = cv2.boxPoints(min_rect)
            points = np.int0(points)  # 最小外接矩形的四个点坐标
            # cv2.drawContours(ori_img, [points], 0, (0, 255, 0), 5)
            # plt_show0(ori_img)

            # 求轮廓的四个顶点
            approx = cv2.approxPolyDP(cnt, 123, True)
            # 轮廓为4个点表示找到纸张
            if len(approx) == 4:
                approx = approx.reshape(4, 2)
                approx_list = approx.tolist()
                point1, point2, point3, point4 = approx_list[0], approx_list[1], approx_list[2], approx_list[3]
                print(approx.tolist())
                docCnt = approx

            side1 = int(min_rect[1][0])
            side2 = int(min_rect[1][1])
            rotate_angle = round(min_rect[2], 2)
            need_rotate = True
            # 车牌没有倾斜
            if side1 < side2 and rotate_angle == 90.0:  # 说明是正的，没有旋转
                new_point1 = point1
                new_point2 = [point1[0], point1[1]+side1]
                new_point3 = [new_point2[0]+side2, new_point2[1]]
            # 正倾斜时，右侧高，side1>side2
            elif side1 < side2 and rotate_angle != 90.0:
                print('zheng')
                temp_width = int(side2/3.14)
                new_point1 = point1
                new_point2 = [point1[0]-side2, point1[1]]
                new_point3 = [new_point2[0], new_point2[1]+temp_width]
                new_point4 = [new_point3[0]+side2, new_point3[1]]
            # 负倾斜，左侧高
            elif side1 >= side2 and rotate_angle != 90.0:
                print('fu')
                temp_width = int(side1/3.14)
                new_point1 = point1
                new_point2 = [point1[0], point1[1] + temp_width]
                new_point3 = [new_point2[0] + side1, new_point2[1]]
                new_point4 = [new_point3[0], new_point3[1] - temp_width]
            # 车牌倾斜90度，上下窄，两侧宽
            elif side1 >= side2 and rotate_angle == 90.0:  # 车辆侧翻情况，或者手机横向拍照
                new_point1 = point1
                new_point2 = [point1[0]+side1, point1[1]]
                new_point3 = [new_point2[0], new_point2[1]-side2]
            else:  # 不符合上述情况，不需要旋转
                new_point1, new_point2, new_point3 = point1, point2, point3
                need_rotate = False
                print('you don\'t need rotate!')

            if need_rotate:
                src_three_points = np.float32([point1, point2, point3])
                dst_three_points = np.float32([new_point1, new_point2, new_point3])
                M = cv2.getAffineTransform(src_three_points, dst_three_points)
                dst = cv2.warpAffine(ori_img, M, (ori_img.shape[1], ori_img.shape[0]))
            else:
                dst = ori_img

            plt_show0(dst)
            print(f"img_name:{img_name},side1:{int(min_rect[1][0])},side2:{int(min_rect[1][1])},rotate_angle:{round(min_rect[2],2)}")


        # 在原图上
        # 分别打印出4个顶点
        # 分别以每个点为圆心，10为半径，(255, 0, 0)为颜色，画圆圈
        # for peak in docCnt:
        #     # peak = peak[0]
        #     cv2.circle(ori_img, tuple(peak), 10, (0, 0, 255), 5)

        # plt_show0(ori_img)


def test_rename_pic():
    pass
    # img_dir = '/data2/lily/datasets/car_plate_test'
    # img_names = os.listdir(img_dir)
    # print(len(img_names))
    # digit_num = len(str(len(img_names))) - 1
    # img_names = tqdm(img_names)
    # for idx, img_name in enumerate(img_names):
    #     suffix = img_name.split('.')[-1]
    #     img_path = os.path.join(img_dir, img_name)
    #     os.rename(img_path, os.path.join(img_dir, str(idx).zfill(digit_num) + '.' + suffix))

    # old_dir = '/data2/lily/datasets/car_plate_test_result/better/old/'
    # new_dir = '/data2/lily/datasets/car_plate_test_result/better/new/'
    # img_dir = '/data2/lily/datasets/car_plate_test_result/better/'
    # old_file_list = os.listdir(old_dir)
    # new_file_list = os.listdir(new_dir)
    # whole_file_len = len(old_file_list) + len(new_file_list)
    # digit_num = len(str(whole_file_len)) - 1
    # for idx in range(whole_file_len):
    #     if idx < len(old_file_list):
    #         img_path = os.path.join(old_dir, old_file_list[idx])
    #     else:
    #         img_path = os.path.join(new_dir, new_file_list[idx - len(old_file_list)])
    #     suffix = img_path.split('.')[-1]
    #     os.rename(img_path, os.path.join(img_dir, str(idx).zfill(digit_num) + '.' + suffix))


def test_hsv_range():
    img_dir = '/data2/lily/datasets/'
    img_names = os.listdir(img_dir)
    img_names = sorted(img_names)
    img_names = ['hsv过滤测试.jpg']
    rgb_list = [[240,5,66], [0,0,58], [240,20,19], [240,42,27],
                [240,55,35], [218,96,50], [225,84,29], [226,84,25],
                [224,60,32], [199,100,88], [212,100,96], [216,96,62],
                [200,100,75], [233,88,41]]
    result = []
    # for i in rgb_list:
    #     rgb = np.array(i)
    #     trans_list = [0.71,2.55,2.55]
    #     trans = np.array(trans_list)
    #     result.append(rgb*trans)
    #     print(rgb*trans)
    # result_new = result[-5:]
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        # 用滤波
        # img = cv2.GaussianBlur(img, (5,5), 0)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lowerb = np.array([135, 145, 60])  # 50可以
        upperb = np.array([165, 255, 255])
        mask = cv2.inRange(img_hsv, lowerb, upperb)
        plt_show(mask)


def useful_ann():
    # for peak in approx:
    #     peak = peak[0]
    #     cv2.circle(ori_img, tuple(peak), 5, (0, 0, 255), 2)
    # plt_show0(ori_img)
    pass


def test_recognition():
    PLATE_CHARS_LETTER = ["A", "B", "C", "D", "E", "F", "G",
                          "H", "J", "K", "L", "M", "N",
                          "P", "Q", "R", "S", "T",
                          "U", "V", "W", "X", "Y", "Z"]
    PLATE_CHARS_NUMS_LETTER = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                          "A", "B", "C", "D", "E", "F", "G",
                          "H", "J", "K", "L", "M", "N",
                          "P", "Q", "R", "S", "T",
                          "U", "V", "W", "X", "Y", "Z"]
    font_ch = '/home/lily/EfficientNet_pytorch/gen-Chinese-plate/font/platechar.ttf'
    font_en = '/home/lily/EfficientNet_pytorch/gen-Chinese-plate/font/platech.ttf'
    fontC = ImageFont.truetype(font_ch, 43, 0)
    fontE = ImageFont.truetype(font_en, 120, 0)
    img_path = '/data2/lily/datasets/car_plate_test_result/plate/00.JPG'
    ori_img = cv2.imread(img_path)
    approx_list = []  # 存储所有字符轮廓
    # plt_show0(ori_img)
    count = 0
    licenses = []
    gray2 = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros_like(gray2)
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    plt_show(gray2)

    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # (9 1)
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))  # (9 7) (13 5)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(gray2, element1, iterations=1)
    # erosion = cv2.erode(erosion, element2, iterations=1)
    ret2, binary2 = cv2.threshold(erosion, 120, 255, cv2.THRESH_BINARY)
    plt_show(binary2)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(binary2, element1, iterations=2)
    # dilation2 = cv2.dilate(dilation2, element2, iterations=2)
    plt_show(dilation2)

    ret2, binary2 = cv2.threshold(dilation2, 120, 255, cv2.THRESH_BINARY)
    plt_show(binary2)

    contours, hierarchy = cv2.findContours(binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy.reshape((hierarchy.shape[1], hierarchy.shape[2]))
    for i in range(len(contours)):
        cnt = contours[i]
        if hierarchy[i][3] == -1:  # 由于在遍历父轮廓的时候对子轮廓已经处理了，所以不处理子轮廓
            area = cv2.contourArea(cnt)  # 计算当前轮廓的面积
            if area > 600:  # 只处理面积大于600的轮廓
                approx = cv2.approxPolyDP(cnt, 3, False)  # 计算当前父轮廓的点集，每3个像素画一个点
                min_rect = cv2.minAreaRect(cnt)  # 计算轮廓的最小外接矩形
                x = min_rect[0][0]  # 轮廓的左上角坐标

                side1 = int(min_rect[1][0])
                side2 = int(min_rect[1][1])
                height, width = (side1, side2) if side1 < side2 else (side2, side1)
                ratio = width / height + 0.001  # 轮廓的长宽之比，加0.001是为了防止除零报错

                if approx.shape[0] >= 3 and ratio > 1.6 and ratio < 2.3 and height > 20:  # 符合条件的才会被筛选进来
                    count += 1  # 记录有多少个符合条件，期望值为6
                    approx = approx.reshape(approx.shape[0], 2)

                    if hierarchy[i][2] != -1:  # 如果当前轮廓的子轮廓存在
                        child_cnt = contours[hierarchy[i][2]]  # 子轮廓的cnt
                        approx_child = cv2.approxPolyDP(child_cnt, 3, False)  # 用点逼近轮廓形状
                        approx_child = approx_child.reshape(approx_child.shape[0], 2)
                        approx = np.vstack((approx, approx_child))  # 将父子轮廓合并

                    approx_list.append([approx, x, min_rect, height, width])  # 将符合条件的轮廓信息添加到一个列表中
                    # cv2.fillPoly(ori_img, [approx], (0,0,255))  # 展示填充轮廓后的效果
                    # cv2.drawContours(ori_img, cnt, -1, (0, 255, 255), 2)  # 以线段方式展示选择的轮廓

                    # plt_show0(ori_img)
                    # print('hello')

    # plt_show0(ori_img)
    print(f"=======count======: {count}")
    approx_list = sorted(approx_list, key=lambda x: x[1])  # 对字符轮廓进行排序，按照横坐标大小，小的在前，大的在后
    index = np.random.randint(0, len(approx_list), 1)[0]
    approx = approx_list[index][0]  # 随机挑选出一个字符轮廓
    char = ''

    cv2.fillPoly(mask_img, [approx], 255)  # 生成要篡改字符的mask
    dst_img = cv2.inpaint(ori_img, mask_img, 3, cv2.INPAINT_TELEA)  # 这个inpainting方法效果更好
    # plt_show0(dst_img)

    min_rect = approx_list[index][2]
    points = cv2.boxPoints(min_rect)
    points = np.int0(points)  # 最小外接矩形的四个点坐标
    points = sorted(points, key=lambda x: x[0] + x[1])
    point1 = points[0]  # 最小外接矩形的左上角坐标
    x, y = point1[0], point1[1]
    width, height = approx_list[index][3], approx_list[index][4]  # 注意：这里把长的当做高，短边当做宽

    if index == 0:  # 只能在字母中找
        index_first = np.random.randint(0, 24, 1)[0]
        char = PLATE_CHARS_LETTER[index_first]
    else:  # 在字母和数字中找 除去O和I
        index_first = np.random.randint(0, 32, 1)[0]
        char = PLATE_CHARS_NUMS_LETTER[index_first]
    print(f"======char======{char}")
    # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGBA)
    temp = GenCh1(fontE, char, width, height)
    # temp_bgr = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
    # plt_show0(temp_bgr)
    # temp_bgr = cv2.resize(temp_bgr, (width, height))
    # plt_show0(temp_bgr)
    # dst_img[y:y + height, x:x + width, :] = temp
    # dst_img[x:x + width, y:y + height] = temp
    # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGBA2BGR)
    for i in range(height):
        for j in range(width):
            for k in range(3):
                if temp[i, j, k] == 0:
                    dst_img[y + i, x + j, k] = 255

    # dst_img2 = cv2.inpaint(ori_img, mask_img, 3, cv2.INPAINT_NS)
    # plt_show(mask_img)
    # plt_show0(dst_img)
    # print(count)
    # print("DONE")

    return dst_img


if __name__ == '__main__':
    img_path = '/home/lily/EfficientNet_pytorch/platenumber/char_img.jpg'
    ori_img = cv2.imread(img_path)
    bg_cor_bgr = [160, 40, 3]
    char = 'B'
    use_tet_gan(ori_img, bg_cor_bgr, char)
    # separate_blue()

    print("DONE")
    pass
