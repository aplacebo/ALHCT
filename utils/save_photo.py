import os
import os.path as osp
import torch
import numpy as np
import cv2

def save_crm(features, output, target, img_paths, path):
    for i in range(features.size(0)):
        # 保存位置
        filename = img_paths[i]
        os.makedirs(path, exist_ok=True)
        if torch.cuda.is_available():
            features = features.cpu()
        for j in range(features.size(1)):
            # 预测类别
            predict_id = torch.argmax(output[i]).item()
            prob = torch.softmax(output[i], dim=0)
            idx = target[i].item()

            classname = img_paths[i].split('/')[-2:]
            file_path = os.path.join(path, classname[0])
            os.makedirs(file_path, exist_ok=True)

            # img_name = classname[1].split('.')[0] + "_channel=" + str(j) + "_idx=" + str(idx) + "_predict=" + str(
            #     predict_id) + "_prob=" + str(prob[idx]) + '.jpg'
            # save_path = osp.join(file_path, img_name)
            img_name_gray = classname[1].split('.')[0] + "_channel=" + str(j) + "_idx=" + str(idx) + "_predict=" + str(
                predict_id) + "_prob=" + str(prob[idx]) + "_gray_" + '.bmp'
            save_path_gray = osp.join(file_path, img_name_gray)
            # 特征图转热力图
            img = features[i][j].detach().numpy()
            # img = cv2.resize(img, (224, 224))
            pmin = np.min(img)
            pmax = np.max(img)
            img = ((img - pmin) / (pmax - pmin + 0.000001)) *255  # float在[0，1]之间，转换成0-255
            img = img.astype(np.uint8)  # 转成unit8
            cv2.imwrite(save_path_gray, img)
            # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
            # # img = img[:, :, ::-1]
            # # 保存图像
            # cv2.imwrite(save_path_gray, img)

def returnCAM(features, weight_softmax, output, target, img_paths, path):
    B, nc, h, w = features.shape
    for i in range(features.size(0)):
        predict_id = torch.argmax(output[i]).item()
        prob = torch.softmax(output[i], dim=0)
        idx = target[i].item()
        os.makedirs(path, exist_ok=True)
        if torch.cuda.is_available():
            features = features.cpu()
        classname = img_paths[i].split('/')[-2:]
        file_path = os.path.join(path, classname[0])
        os.makedirs(file_path, exist_ok=True)

        img_name = classname[1].split('.')[0] + "_idx=" + str(idx) + "_predict=" + str(
            predict_id) + "_prob=" + str(prob[idx]) + '.bmp'
        save_path = osp.join(file_path, img_name)
        img_name_gray = classname[1].split('.')[0]  + "_idx=" + str(idx) + "_predict=" + str(
            predict_id) + "_prob=" + str(prob[idx]) + "_gray_" + '.bmp'
        save_path_gray = osp.join(file_path, img_name_gray)
        cam = weight_softmax[predict_id].dot(features[i,:,:,:].reshape((nc, h * w)).detach().cpu())
        # 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
        cam = cam.reshape(h, w)# 得到单张特征图
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        # cam_img = cam_img.astype(np.uint8)  # 转成unit8
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, (224, 224))
        cv2.imwrite(save_path_gray, cam_img)
        cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)  # 生成heat map
        # img = img[:, :, ::-1]
        # 保存图像
        cv2.imwrite(save_path, cam_img)
    # # 类激活图上采样到 256 x 256
    # size_upsample = (256, 256)
    #
    # output_cam = []
    # # 将权重赋给卷积层：这里的weigh_softmax.shape为(1000, 512)
    # # 				feature_conv.shape为(1, 512, 13, 13)
    # # weight_softmax[class_idx]由于只选择了一个类别的权重，所以为(1, 512)
    # # feature_conv.reshape((nc, h * w))后feature_conv.shape为(512, 169)
    # cam = weight_softmax[class_idx].dot(features.reshape((nc, h * w)).detach().cpu())
    # # 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
    # cam = cam.reshape(h, w) # 得到单张特征图
    # # 特征图上所有元素归一化到 0-1
    # cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    # # 再将元素更改到　0-255
    # cam_img = np.uint8(255 * cam_img)
    # output_cam.append(cv2.resize(cam_img, size_upsample))
    # return output_cam