import os
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import load_model
from keras.models import Model
from resnet import ResnetBuilder
import nibabel as nib
import cv2,xlwt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]
def random_shuffle(data,label):
    randnum = np.random.randint(0, 1234)
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data,label
def process_test(path1,path2):
    x_paths = []
    y_labels = []
    image_paths = os.listdir(path1)
    for path_img in image_paths:
        if path_img.endswith(".hdr"):
            continue
        x_paths.append(path1 + "/" + path_img)
        y_labels.append(1)
    image_paths = os.listdir(path2)
    for path_img in image_paths:
        if path_img.endswith(".hdr"):
            continue
        x_paths.append(path2 + "/" + path_img)
        y_labels.append(0)
    random_shuffle(x_paths, y_labels)
    batch_res = []
    y_res = []
    for i in range(len(x_paths)):
        batch_res.append(read_single(x_paths[i]))
        y_res.append(y_labels[i])
    y_res = np.array(y_res)
    y = (np.arange(2) == y_res[:, None]).astype(int)
    return np.array(batch_res),y
def process_image(path1,path2,batchsize=None):
    x_paths = []
    y_labels = []
    image_paths = os.listdir(path1)
    for path_img in image_paths:
        if path_img.endswith(".hdr"):
            continue
        x_paths.append(path1+"/"+path_img)
        y_labels.append(1)
    image_paths = os.listdir(path2)
    for path_img in image_paths:
        if path_img.endswith(".hdr"):
            continue
        x_paths.append(path2+"/"+path_img)
        y_labels.append(0)
    random_shuffle(x_paths,y_labels)
    count = 0

    while(count+batchsize<len(x_paths)):
        batch_res = []
        y_res = []
        for index in range(0,batchsize):
            batch_res.append(read_single(x_paths[count+index]))
            y_res.append(y_labels[count+index])
        count += batchsize
        y_res = np.array(y_res)
        y = (np.arange(2) == y_res[:, None]).astype(int)
        yield np.array(batch_res),y

def read_single(img_path):
    img = nib.load(img_path).get_data()
    img[img < 0] = 0
    img = np.transpose(img, [2, 1, 0])

    start = 10
    img_full_9 = np.array(img[start])
    for slice in range(1, 9):
        img_full_9 = np.concatenate((img_full_9, img[start + slice]), axis=1)
    start += 5
    img_full = img_full_9
    for hang in range(1,9):
        img_full_9 = np.array(img[start])
        for slice in range(1,9):
            img_full_9 = np.concatenate((img_full_9,img[start+slice]),axis = 1)
        start += 5
        img_full = np.concatenate((img_full,img_full_9),axis=0)

    img_full = cv2.resize(np.array(img_full),(224, 224))
    img_full = (img_full - np.min(img_full)) / (np.max(img_full) - np.min(img_full))
    img_full = np.expand_dims(img_full, axis=0)
    img_full = np.concatenate((img_full, img_full, img_full), axis=0)
    # plt.imshow(np.array(img_full))
    # plt.waitforbuttonpress()
    return img_full

def ptint_out_in_sheet(dense1_output):
    for obj_index in range(len(dense1_output)):
        for index, item in enumerate(dense1_output[obj_index]):
            sheet.write(index, obj_index, str(item))
    f.save("./out.xls")

def model_compile(model):
    K.set_image_data_format('channels_first')
    model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer="sgd")
    # assert True, "Failed to compile with '{}' dim ordering".format(ordering)
def calculate_metric(gt, pred):
    gt2 = []
    pred2 = []
    for i in range(len(pred)):
        pred2.append(0 if pred[i,0]>pred[i,1] else 1)
        gt2.append(0 if gt[i,0]>gt[i,1] else 1)
    confusion = confusion_matrix(gt2,pred2)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity:',TP / float(TP+FN))
    print('Specificity:',TN / float(TN+FP))
    return (TP+TN)/float(TP+TN+FP+FN),TP / float(TP+FN),TN / float(TN+FP)
if __name__=='__main__':
    K.set_image_data_format('channels_first')
    k_total = 10 # 交叉验证次数
    f = xlwt.Workbook()
    sheet = f.add_sheet('yourname', cell_overwrite_ok=True)
    for k_count in range(k_total):
        accuracy_max = 0
        # 搭建模型
        model= ResnetBuilder.build_resnet_34((3, 224, 224), 2)
        model_compile(model)
        # X，Y 为数据集，根据需要修改路径和预处理方式
        X, Y= process_test("./Data/test/AD","./Data/test/NC")
        #k折交叉验证
        skf = StratifiedKFold(n_splits=5)  # 折数可改
        for cnt, (train, test) in enumerate(skf.split(X, Y)):
            # for (batch,batchlabel) in process_image("./Data/test/AD",
            #                            "./Data/test/NC",4):
            #     model.fit(batch, batchlabel, epochs=1, validation_split=0.1)
            model.fit(X[train],Y[test],epochs=100,validation_split=0.2)
            pred = model.predict(np.array(X[test]))
            accuracy, sensitivity, specificity = calculate_metric(Y[test], pred)
            # 在准确率较高的时候保存模型参数,也可以每次都保存
            if(accuracy_max<accuracy):
                model.save('./my_model.h5')
                accuracy_max = accuracy
                print("**************           accuracy: {}".format(accuracy))
                print("**************           sensitivity: {}".format(sensitivity))
                print("**************           specificity: {}".format(specificity))

            feature_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer('flatten_1').output)
            #最后分类器前一层的输出
            dense1_output = feature_layer_model.predict(X[test])
            #每次会覆盖上一次的
            ptint_out_in_sheet(dense1_output)
