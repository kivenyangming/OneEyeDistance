import cv2
import numpy as np


class CamSet():
    def __init__(self):
        # 单目相机参数
        # 内参矩阵
        self.cameraMatrix = np.array([[2049.608299, 1.241852862, 1032.391255],
                                      [0, 2066.791362, 550.6131349],
                                      [0, 0, 1]])

        # 相机畸变系数矩阵，5*1矩阵(k1,k2,p1,p2,k3)0.108221558	-0.232697802
        self.distCoeffs = np.array([0.108221558, -0.232697802, 0.002050653, -0.004714754, 0])

    def CorrectImage(self, img):
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, (w, h), 1, (w, h), 0)
        # 计算无畸变和修正转换关系
        mapx, mapy = cv2.initUndistortRectifyMap(self.cameraMatrix, self.distCoeffs, None, newCameraMatrix, (w, h),
                                                 cv2.CV_16SC2)
        # 重映射 输入是矫正后的图像
        CorrectImg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        return CorrectImg
