# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ai_image_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout, QTextBrowser, QFrame
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QPixmap
#from PyQt5.QtGui import QIcon, QPixmap
import tensorflow as tf
from nets import vgg_avgpool as vgg
import scipy # pip install scipy
import os, sys
import numpy as np
import cv2

try:
    tf.reset_default_graph()
except:
    pass


RGB_process = np.array([123.68, 116.78, 103.94]).reshape((1,1,1,3))

#합성을 위한 이미지 크기 및 weight 값 설정
height = 500# 500
width = 500#500
content_weight= 0.0001
style_weight = 10000000

store_path = "./taehee_store/"
# store_path 생성
if not os.path.exists(store_path):
    os.makedirs(store_path)


#gui 구성
class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        #self.textBrowser.setGeometry(QtCore.QRect(55, 391, 671, 71))
        #self.textBrowser.setObjectName("textBrowser")
        
        #self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        #self.progressBar.setGeometry(QtCore.QRect(150, 500, 501, 23))
        #self.progressBar.setProperty("value", 24)
        #self.progressBar.setObjectName("progressBar")
        
        #self.graphicsView = QtWidgets.QVBoxLayout(self.centralwidget)
        self.graphicsView = QtWidgets.QLabel(self.centralwidget)
        #self.label.setGeometry(QtCore.QRect(50, 120, 256, 192))
        self.graphicsView.setGeometry(QtCore.QRect(50, 120, 256, 192))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setStyleSheet("background-color: white;")
        self.graphicsView.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        #self.graphicsView.addWidget(self.label)

        # self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        # self.graphicsView.setGeometry(QtCore.QRect(50, 120, 256, 192))
        # self.graphicsView.setObjectName("graphicsView")
        # print(dir(self.graphicsView))

        #self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2 = QtWidgets.QLabel(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(360, 120, 256, 192))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_2.setStyleSheet("background-color: white;")
        self.graphicsView_2.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(670, 90, 93, 100))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.on_click_merge)
        
        self.pushButton_mosaic = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_mosaic.setGeometry(QtCore.QRect(670, 230, 93, 100))
        self.pushButton_mosaic.setObjectName("pushButton_mosaic")
        self.pushButton_mosaic.clicked.connect(self.on_click_merge_mosaic)

        #self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        #self.horizontalSlider.setGeometry(QtCore.QRect(100, 80, 160, 22))
        #self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        #self.horizontalSlider.setObjectName("horizontalSlider")
        
        #self.horizontalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        #self.horizontalSlider_2.setGeometry(QtCore.QRect(410, 80, 160, 22))
        #self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        #self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.con_label = QtWidgets.QLabel(self.centralwidget)
        self.con_label.setGeometry(QtCore.QRect(120, 80, 160, 22))
        self.con_label.setObjectName("con_label")

        self.style_label = QtWidgets.QLabel(self.centralwidget)
        self.style_label.setGeometry(QtCore.QRect(430, 80, 160, 22))
        self.style_label.setObjectName("style_label")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(260, 320, 41, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.on_click_content_file)

        self.textBrowser_con = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_con.setGeometry(QtCore.QRect(80, 320, 170, 31))
        self.textBrowser_con.setObjectName("textBrowser_con")
        
        self.textBrowser_style = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_style.setGeometry(QtCore.QRect(390, 320, 170, 31))
        self.textBrowser_style.setObjectName("textBrowser_style")
        
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 20, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.on_click_camera)

        # 이미지 자르기 
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(100, 400, 150, 100))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.on_click_image_crop_and_style_transfer)

        self.pushButton_two_styles = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_two_styles.setGeometry(QtCore.QRect(400, 400, 150, 100))
        self.pushButton_two_styles.setObjectName("pushButton_two_styles")
        self.pushButton_two_styles.clicked.connect(self.on_click_image_two_style_transfer)
        #self.pushButton_two_styles.clicked.connect()
        
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(570, 320, 41, 28))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(self.on_click_style_file)

        self.quitButton = QtWidgets.QPushButton(self.centralwidget)
        self.quitButton.setGeometry(QtCore.QRect(620, 510, 141, 28))
        self.quitButton.setObjectName("quitButton")
        self.quitButton.clicked.connect(QCoreApplication.instance().quit)   

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "이미지변환"))
        self.pushButton_mosaic.setText(_translate("MainWindow", "모자이크\n이미지변환"))
        self.pushButton_2.setText(_translate("MainWindow", "파일"))
        self.pushButton_4.setText(_translate("MainWindow", "카메라 촬영"))
        self.pushButton_5.setText(_translate("MainWindow", "이미지\n자르기"))
        self.pushButton_6.setText(_translate("MainWindow", "파일"))
        self.quitButton.setText(_translate("MainWindow","나가기"))
        self.con_label.setText(_translate("MainWindow", "컨텐츠 이미지"))
        self.style_label.setText(_translate("MainWindow", "스타일 이미지"))
        self.pushButton_two_styles.setText(_translate("MainWindow", "두가지\n스타일변환"))

    #콘텐츠 파일 불러오기
    def on_click_content_file(self):
        self.content_img_path = QtWidgets.QFileDialog.getOpenFileName()[0]
        print('content_img_path', self.content_img_path)
        #pixmap = QtGui.QPixmap(self.content_img_path)
        #pixmap_resize = pixmap.scaled(142, 136, QtCore.Qt.KeepAspectRatio)
        #height, width = self.graphicsView.sizeHint().height(), self.graphicsView.sizeHint().width()
        #print(height, width)
        #self.graphicsView.setPixmap(QtGui.QPixmap(pixmap_resize))
        pixmap = QPixmap(self.content_img_path)
       # self.label = QLabel(self)
        #self.graphicsView.setGeometry(30, 40, 100, 150)
        self.scaledImg = pixmap.scaled(self.graphicsView.size(), Qt.KeepAspectRatio)
        self.graphicsView.setPixmap(self.scaledImg)
        self.textBrowser_con.append(self.content_img_path)

    #스타일 파일 불러오기
    def on_click_style_file(self):
        self.style_img_path = QtWidgets.QFileDialog.getOpenFileName()[0]
        print('style_img_path', self.style_img_path)
        pixmap = QPixmap(self.style_img_path)
        self.scaledImg_2 = pixmap.scaled(self.graphicsView_2.size(), Qt.KeepAspectRatio)
        self.graphicsView_2.setPixmap(self.scaledImg_2)
        self.textBrowser_style.append(self.style_img_path)

    #이미지 변환 모델 실행
    def on_click_merge(self):
        style_transfer(self.style_img_path, self.content_img_path, height=800, width=800, mosaic_mode=False, name='original')
    #이미지 모자이크 변환 모델 실행
    def on_click_merge_mosaic(self):
        style_transfer(self.style_img_path, self.content_img_path, height=800, width=800, mosaic_mode=True)
    
    #두가지 스타일 적용
    def on_click_image_two_style_transfer(self):
        self.content_img_path = QtWidgets.QFileDialog.getOpenFileName(caption='content_img 열기')[0]
        print('content_img_path', self.content_img_path)
        self.style_img_path = QtWidgets.QFileDialog.getOpenFileName(caption='style_img_1 열기')[0]
        print('style_img_path', self.style_img_path)
        self.style_img_path2 = QtWidgets.QFileDialog.getOpenFileName(caption='style_img_2 열기')[0]
        print('style_img_path2', self.style_img_path2)
        style_transfer(self.style_img_path, self.content_img_path, style_img_path2=self.style_img_path2, height=800, width=800, mosaic_mode=False, name='two_style')

    #이미지 잘라서 모델 적용
    def on_click_image_crop_and_style_transfer(self):
        self.content_img_for_crop_path = QtWidgets.QFileDialog.getOpenFileName(caption='content_img 열기')[0]
        print('content_img_for_crop_path', self.content_img_for_crop_path)

        self.cropping = False
        self.x_start, self.y_start, self.x_end, self.y_end = -1, -1, -1, -1
        with open(self.content_img_for_crop_path, 'rb') as image_byte:
            image = bytearray(image_byte.read())
            image = np.array(image, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, dsize=(800, 800), interpolation=cv2.INTER_AREA)

        #마우스 드래그를 이용하여 이미지 잘라내기
        def mouse_crop(event, x, y, flags, param):
            # grab references to the global variables
            #global x_start, y_start, x_end, y_end, cropping
         
            # if the left mouse button was DOWN, start RECORDING
            # (x, y) coordinates and indicate that cropping is being
            if event == cv2.EVENT_LBUTTONDOWN:
                self.x_start, self.y_start, self.x_end, self.y_end = x, y, x, y
         
            # Mouse is Moving
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.x_start and self.y_start:
                    self.x_end, self.y_end = x, y
         
            # if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates
                self.x_end, self.y_end = x, y
                #self.cropping = True#False # cropping is finished
         
                refPoint = [(self.x_start, self.y_start), (self.x_end, self.y_end)]
                cropped_content_img_path = [str(self.x_start), str(self.y_start), str(self.x_end), str(self.y_end)]
                self.cropped_content_img_path = '_'.join(cropped_content_img_path) +'.png'
                print(self.cropped_content_img_path)
                
                #test = image[input[0], input[1], :]
                #test = image.copy()

                if len(refPoint) == 2: #when two points were found
                    self.roi = image[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                    cv2.imshow("Cropped", self.roi)
                    cv2.imwrite(self.cropped_content_img_path, self.roi)

                self.cropping = True

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_crop)


        #잘라내기 저장 및 종료
        while True:        
            k = cv2.waitKey(1)

            if not self.cropping:
                cv2.imshow("image", image)
         
            elif self.cropping:
                # ESC 버튼 입력시 종료
                if k%256 == 27:
                    print("Escape hit, closing...")
                    break
                
        cv2.destroyAllWindows()
        
        height, width, _ = self.roi.shape
        self.style_img_path = QtWidgets.QFileDialog.getOpenFileName(caption='style_img 열기')[0]
        print('style_img_path', self.style_img_path)
        # style_transfer
        style_transfer(self.style_img_path, self.cropped_content_img_path, height=height, width=width, mosaic_mode=False, name='cropped_image')
        
        # style transfer된 이미지 읽기.
        store_path = "./taehee_store/"
        cropped_trasnfer_img_path = os.path.join(store_path, 'cropped_image.jpg') # 2=> 500
        with open(cropped_trasnfer_img_path, 'rb') as cropped_transfer_img_byte:
            cropped_transfer_img = bytearray(cropped_transfer_img_byte.read())
            cropped_transfer_img = np.array(cropped_transfer_img, np.uint8)
            cropped_transfer_img = cv2.imdecode(cropped_transfer_img, cv2.IMREAD_UNCHANGED)
        # remove cropped_transfer_img
        os.remove(cropped_trasnfer_img_path)

        # cropped image merge
        cordinates = self.cropped_content_img_path.split('.')[0].split('_')
        cordinates = [int(i) for i in cordinates]
        x_start, y_start, x_end, y_end = cordinates
        print(x_start, y_start, x_end, y_end)


        image[y_start:y_end, x_start:x_end, :] = cropped_transfer_img
        cv2.imwrite(os.path.join(store_path, 'crop_result'+'.jpg'), image)

    #노트북 웹캠과 연결
    def on_click_camera(self):
        capture = cv2.VideoCapture(0)
        cv2.namedWindow("camera")
        img_counter = 0

        while True:
            ret, frame = capture.read()
            cv2.imshow("camera", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "C:\myphoto{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        capture.release()
        cv2.destroyAllWindows()

    def con_label(self):
        self.con_label.setText(test)


    #def image_crop(self):






#사진파일 읽어오기
def read_image(path, height, width):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(height, width)) 
    img = tf.keras.preprocessing.image.img_to_array(img) #img = height,width,3
    img = np.array([img]) # [0, 255]
    img -= RGB_process
    return img
#gram_matrix 계산
def gram_matrix(conv):
    # conv: [height, width, channel]
    conv_transpose = tf.transpose(conv, (2, 0, 1)) # [channel, height, width]
    conv_flat = tf.layers.flatten(conv_transpose) # [channel, height*width]
    conv_NM = tf.reduce_prod(conv_flat.get_shape()) # 논문 N*M
    gram = tf.matmul(conv_flat, tf.transpose(conv_flat))
    return gram, tf.cast(conv_NM, tf.float32)
  
def make_vgg19(X, height, width):
    score, end_points= vgg.vgg_19(inputs = X, num_classes=0)
    dic = {}

    # for style
    conv1_1 = end_points['vgg_19/conv1/conv1_1'] # [1, 500, 500, 64]
    conv1_1 = conv1_1[0, :, :, :] # [500, 500, 64]
    dic['conv1_1'] = conv1_1

    conv2_1 = end_points['vgg_19/conv2/conv2_1'] # [1, 250, 250, 128]
    conv2_1 = conv2_1[0, :, :, :] # [250, 250, 128]
    dic['conv2_1'] = conv2_1

    conv3_1 = end_points['vgg_19/conv3/conv3_1'] # [1, 125, 125, 256]
    conv3_1 = conv3_1[0, :, :, :] # [125, 125, 256]
    dic['conv3_1'] = conv3_1
  
    conv4_1 = end_points['vgg_19/conv4/conv4_1'] # [1, 62, 62, 512]
    conv4_1 = conv4_1[0, :, :, :] # [62, 62, 512]
    dic['conv4_1'] = conv4_1

    conv5_1 = end_points['vgg_19/conv5/conv5_1'] # [1, 31, 31, 512]
    conv5_1 = conv5_1[0, :, :, :] # [31, 31, 512]
    dic['conv5_1'] = conv5_1

    # for content
    conv4_2 = end_points['vgg_19/conv4/conv4_2'] # [1, 62, 62, 512]
    conv4_2 = conv4_2[0, :, :, :] # [62, 62, 512]
    dic['conv4_2'] = conv4_2
    return dic


#모자이크 생성을 위한 위치 입력
def mosaic(height, width, num=20):
    coordinate_list = []
    height_min_mosaic_length = max(int(height*0.05), 10)
    height_max_mosaic_length = max(int(height*0.15), 10)
    width_min_mosaic_length = max(int(width*0.05), 10)
    width_max_mosaic_length = max(int(width*0.15), 10)
    
    #랜덤 좌표 입력
    for _ in range(num):
        height_coordinate = np.random.randint(0, height-height_max_mosaic_length)
        width_coordinate = np.random.randint(0, width-width_max_mosaic_length)      
        height_length = np.random.randint(height_min_mosaic_length, height_max_mosaic_length+1)
        width_length = np.random.randint(width_min_mosaic_length, width_max_mosaic_length+1)
        coordinate_list.append([height_coordinate, width_coordinate, height_length, width_length])

    return coordinate_list

#이미지 변환 모델 실행
def style_transfer(style_img_path, content_img_path, style_img_path2=None, height=800, width=800, mosaic_mode=False, name=''):
    try:
        tf.reset_default_graph()
    except:
        pass

    style_img = read_image(style_img_path, height=height, width=width)
    content_img = read_image(content_img_path, height=height, width=width)
    if style_img_path2:#두가지 스타일 이미지 입력시 실행
        style_img2 = read_image(style_img_path2, height=height, width=width)

    # VGG Model
    X = tf.placeholder(dtype = tf.float32, shape = [None, height, width, 3])
    model_dict = make_vgg19(X, height, width)
    conv4_2 = model_dict['conv4_2'] 
    conv1_1 = model_dict['conv1_1']
    conv2_1 = model_dict['conv2_1']
    conv3_1 = model_dict['conv3_1']
    conv4_1 = model_dict['conv4_1']
    conv5_1 = model_dict['conv5_1']

    # gram
    conv1_1_gram, _ = gram_matrix(conv1_1)
    conv2_1_gram, _ = gram_matrix(conv2_1)
    conv3_1_gram, _ = gram_matrix(conv3_1)
    conv4_1_gram, _ = gram_matrix(conv4_1)
    conv5_1_gram, _ = gram_matrix(conv5_1)


    # restore
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess = sess, save_path = './nets/vgg_19/vgg_19.ckpt')

    # content, style convolution 값 연산.
    content_conv4_2 = sess.run(conv4_2, {X: content_img})
    style_conv1_1_gram, style_conv2_1_gram, style_conv3_1_gram, style_conv4_1_gram, style_conv5_1_gram = sess.run(
            [conv1_1_gram, conv2_1_gram, conv3_1_gram, conv4_1_gram, conv5_1_gram], 
            {X:style_img}
        )
    if style_img_path2:
        style_conv1_1_gram2, style_conv2_1_gram2, style_conv3_1_gram2, style_conv4_1_gram2, style_conv5_1_gram2 = sess.run(
                [conv1_1_gram, conv2_1_gram, conv3_1_gram, conv4_1_gram, conv5_1_gram], 
                {X:style_img2}
            )
    sess.close()
    tf.reset_default_graph()




    ####################################################
    ## 합성 부분 ##

    # neural style transfer
    with tf.variable_scope('noise_img') as scope: #noise image 생성
        noise_img = tf.Variable(tf.constant(content_img), name='noise') #noise image 기본 값을 content image로 지정


    # VGG Model
    noise_model_dict = make_vgg19(noise_img, height, width)
    vgg_variables_to_restore = [i for i in tf.contrib.slim.get_variables_to_restore() if 'noise' not in i.name]

    noise_conv4_2 = noise_model_dict['conv4_2'] 
    noise_conv1_1 = noise_model_dict['conv1_1']
    noise_conv2_1 = noise_model_dict['conv2_1']
    noise_conv3_1 = noise_model_dict['conv3_1']
    noise_conv4_1 = noise_model_dict['conv4_1']
    noise_conv5_1 = noise_model_dict['conv5_1']

    # gram
    noise_conv1_1_gram, noise_conv1_1_NM = gram_matrix(noise_conv1_1)
    noise_conv2_1_gram, noise_conv2_1_NM = gram_matrix(noise_conv2_1)
    noise_conv3_1_gram, noise_conv3_1_NM = gram_matrix(noise_conv3_1)
    noise_conv4_1_gram, noise_conv4_1_NM = gram_matrix(noise_conv4_1)
    noise_conv5_1_gram, noise_conv5_1_NM = gram_matrix(noise_conv5_1)



    # content loss
    content_loss = tf.reduce_sum( (content_conv4_2 - noise_conv4_2)**2 ) * 0.5


    # style loss
    style_loss_1_1 = tf.reduce_sum( tf.square(style_conv1_1_gram - noise_conv1_1_gram) ) / noise_conv1_1_NM / noise_conv1_1_NM / 4 / 5
    style_loss_2_1 = tf.reduce_sum( tf.square(style_conv2_1_gram - noise_conv2_1_gram) ) / noise_conv2_1_NM / noise_conv2_1_NM / 4 / 5
    style_loss_3_1 = tf.reduce_sum( tf.square(style_conv3_1_gram - noise_conv3_1_gram) ) / noise_conv3_1_NM / noise_conv3_1_NM / 4 / 5
    style_loss_4_1 = tf.reduce_sum( tf.square(style_conv4_1_gram - noise_conv4_1_gram) ) / noise_conv4_1_NM / noise_conv4_1_NM / 4 / 5
    style_loss_5_1 = tf.reduce_sum( tf.square(style_conv5_1_gram - noise_conv5_1_gram) ) / noise_conv5_1_NM / noise_conv5_1_NM / 4 / 5
    style_loss = (style_loss_1_1 + style_loss_2_1 + style_loss_3_1 + style_loss_4_1 + style_loss_5_1)

    if style_img_path2:
        style_loss_1_1_2 = tf.reduce_sum( tf.square(style_conv1_1_gram2 - noise_conv1_1_gram) ) / noise_conv1_1_NM / noise_conv1_1_NM / 4 / 5
        style_loss_2_1_2 = tf.reduce_sum( tf.square(style_conv2_1_gram2 - noise_conv2_1_gram) ) / noise_conv2_1_NM / noise_conv2_1_NM / 4 / 5
        style_loss_3_1_2 = tf.reduce_sum( tf.square(style_conv3_1_gram2 - noise_conv3_1_gram) ) / noise_conv3_1_NM / noise_conv3_1_NM / 4 / 5
        style_loss_4_1_2 = tf.reduce_sum( tf.square(style_conv4_1_gram2 - noise_conv4_1_gram) ) / noise_conv4_1_NM / noise_conv4_1_NM / 4 / 5
        style_loss_5_1_2 = tf.reduce_sum( tf.square(style_conv5_1_gram2 - noise_conv5_1_gram) ) / noise_conv5_1_NM / noise_conv5_1_NM / 4 / 5
        style_loss2 = (style_loss_1_1_2 + style_loss_2_1_2 + style_loss_3_1_2 + style_loss_4_1_2 + style_loss_5_1_2)



    # total loss
    total_loss = content_loss * content_weight + style_loss * style_weight

    if style_img_path2:
        total_loss += style_loss2 * style_weight

    # optimizer
    for_train_noise_img = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'noise_img')
    optimizer = tf.train.AdamOptimizer(10) # 학습률 10llss
    minimize = optimizer.minimize(total_loss, var_list=for_train_noise_img) #vgg ,네트워크는 고정하고 noise_img만 학습.

    # sess, restore
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(vgg_variables_to_restore)
    saver.restore(sess = sess, save_path = './nets/vgg_19/vgg_19.ckpt')
    #print(sess.run(tf.reduce_prod(noise_conv1_1_gram[1])))


    if mosaic_mode:
        mosaic_list = mosaic(height, width, num=60)
        mosaic_img = (content_img.copy() + RGB_process)[0]


    for epoch in range(1, 501): # 3 => 501
        img, _, loss = sess.run([noise_img, minimize, total_loss])
        #if epoch % 50 == 0 or epoch == 1:
        if epoch % 50 == 0 or epoch == 1:
        #if epoch == 500:
            print(epoch, loss)
            img += RGB_process
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.reshape(img, [height, width, 3])

            if mosaic_mode:
                for mosaic_info in mosaic_list:
                    height_coordinate, width_coordinate, height_length, width_length = mosaic_info
                    mosaic_img[height_coordinate:height_coordinate+height_length, width_coordinate:width_coordinate+width_length, :] = img[height_coordinate:height_coordinate+height_length, width_coordinate:width_coordinate+width_length, :] 
                scipy.misc.imsave(os.path.join(store_path, 'mosaic.jpg'), mosaic_img)
            else:
                #이미지 저장.
                scipy.misc.imsave(os.path.join(store_path, name+'.jpg'), img)




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

