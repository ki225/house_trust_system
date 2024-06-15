# y-shape detection
使用 y-shape-detection.py 內第十九行的指令，將 data/y-shape 裡的圖片依照比例分成 train、validation 和 test 三者，所以在執行模型時不需要執行第十九行。

後續程式使用 VGG16、MobileNetV2、ResNet50、InceptionV3、EfficientNetB0 這五個常見的模型去做分類，並且生成訓練結果的折線圖。