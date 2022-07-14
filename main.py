import numpy as np

from Config import CamSet
import cv2


def FindCoin(CoinImg):
    # 将图像转换为灰度图像
    gray_img = cv2.cvtColor(CoinImg, cv2.COLOR_BGR2GRAY)
    # 高斯滤波降噪
    gaussian_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    # 利用Canny进行边缘检测
    edges_img = cv2.Canny(gaussian_img, 80, 180, apertureSize=3)
    # 自动检测圆
    # 这里用来检测圆
    circles1 = cv2.HoughCircles(edges_img, cv2.HOUGH_GRADIENT, 1, 1000, param1=300, param2=10, minRadius=50,
                                maxRadius=70)
    if circles1 is None:
        pass
    else:
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(CoinImg, (i[0], i[1]), i[2], (0, 0, 255), 2)
            # print(i[0], i[1], i[2])
            D = 2 * i[2] # 实体圆在图像中占像素的高或宽（此时都为直径）
            FX = 2049.608299  # 测得相机的焦距
            Face_x = 0.0542222  # 实体圆的直径

            FY = 2066.791362  # 测得相机的焦距
            Face_y = 0.054222  # 实体圆的直径

            Distancex = (FX * Face_x) / D  # 计算宽比例得到的距离
            Distancey = (FY * Face_y) / D  # 计算高比例得到的距离

            Distance = (Distancex + Distancey) / 2  # 距离均值
            cv2.putText(CoinImg, "Distance= %.2f" % (Distance), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    # 此处设置视频画面宽高需要和做相机标定时保持一致
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置视频宽度
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置视频高度
    while (True):
        ref, frame = capture.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 进行检测
        Config = CamSet()
        CorrectImg = Config.CorrectImage(frame)
        FindCoin(CorrectImg)
        frame = cv2.cvtColor(CorrectImg, cv2.COLOR_RGB2BGR)
        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
    capture.release()
    cv2.destroyAllWindows()
