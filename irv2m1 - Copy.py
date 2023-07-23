{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "from distutils.dir_util import copy_tree, remove_tree\n",
    "from random import randint\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import matthews_corrcoef as MCC\n",
    "from sklearn.metrics import balanced_accuracy_score as BAS\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import tensorflow_addons as tfa\n",
    "from tensorflow.keras.layers import Dense,Dropout,Flatten,BatchNormalization,GlobalAveragePooling2D\n",
    "from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG\n",
    "import cv2\n",
    "import keras\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "base='./data2/'\n",
    "# root_dir='./'\n",
    "test=base+\"test/\"\n",
    "train=base+\"train/\"\n",
    "wdr=\"./dataset4/\"\n",
    "if os.path.exists(wdr):\n",
    "    remove_tree(wdr)\n",
    "os.mkdir(wdr)\n",
    "copy_tree(train,wdr)\n",
    "copy_tree(test,wdr)\n",
    "print(\"wdc:\",os.listdir(wdr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "work='./dataset4/'\n",
    "classes=['glioma', 'meningioma', 'notumor', 'pituitary']\n",
    "idm=176\n",
    "ida=[176,208]\n",
    "DIM=(idm,idm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data_gen = IDG(rescale=1./255)\n",
    "# train_data_gen=train_data_gen.flow_from_directory(directory=work,target_size=DIM,batch_size=6500)\n",
    "train_data_gen = train_data_gen.flow_from_directory(directory=work, target_size=DIM, batch_size=6500 * 6) #12 for more good res"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show_images(generator):\n",
    "    labels=dict(zip([0,1,2,3],classes))\n",
    "    x,y=generator.next()\n",
    "    plt.figure(figsize=(5,5))\n",
    "    for i in range(2):\n",
    "        ax=plt.subplot(2,2,i+1)\n",
    "        idx=randint(0,200)\n",
    "        plt.imshow(x[idx])\n",
    "        plt.axis(\"off\")\n",
    "        plt.title(\"class:{}\".format(labels[np.argmax(y[idx])]))\n",
    "\n",
    "show_images(train_data_gen)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data,train_labels=train_data_gen.next()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sm=SMOTE(random_state=42)\n",
    "train_data,train_labels=sm.fit_resample(train_data.reshape(-1,idm*idm*3),train_labels)\n",
    "train_data=train_data.reshape(-1,idm,idm,3)\n",
    "print(train_data.shape,train_labels.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data,test_data,train_labels,test_labels=train_test_split(train_data,train_labels,test_size=0.2,random_state=42)\n",
    "train_data,val_data,train_labels,val_labels=train_test_split(train_data,train_labels,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "incmodel=InceptionResNetV2(input_shape=(176,176,3),include_top=False,weights=\"imagenet\")\n",
    "for layer in incmodel.layers:\n",
    "    layer.trainable=False\n",
    "incmodel.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "custommodel=tf.keras.Sequential([incmodel,Dropout(0.10),\n",
    "                            GlobalAveragePooling2D(),Flatten(),\n",
    "                            BatchNormalization(),\n",
    "                            Dense(512,activation='relu'),\n",
    "                            BatchNormalization(),\n",
    "                            Dropout(0.10),\n",
    "                            Dense(256,activation='relu'),\n",
    "                            BatchNormalization(),\n",
    "                            Dropout(0.10),\n",
    "                            Dense(128,activation='relu'),\n",
    "                            BatchNormalization(),\n",
    "                            Dropout(0.10),\n",
    "                            Dense(64,activation='relu'),\n",
    "                            BatchNormalization(),\n",
    "                            Dropout(0.10),\n",
    "                            Dense(4,activation='softmax'),\n",
    "                            ],name=\"inception_cnn_model\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "METRICS=[tf.keras.metrics.CategoricalAccuracy(name='train_acc'),\n",
    "         tf.keras.metrics.AUC(name='val_acc'),\n",
    "         tfa.metrics.F1Score(num_classes=4)]\n",
    "custommodel.compile(loss=tf.losses.CategoricalCrossentropy(),metrics=METRICS)\n",
    "custommodel.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "EPOCHS = 10\n",
    "history = custommodel.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=EPOCHS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig ,ax=plt.subplots(1,3,figsize=(30,5))\n",
    "ax=ax.ravel()\n",
    "for i,metric in enumerate([\"train_acc\",\"val_acc\",\"loss\"]):\n",
    "    ax[i].plot(history.history[metric])\n",
    "    ax[i].plot(history.history[\"val_\"+metric])\n",
    "    ax[i].set_title(\"Model {}\".format(metric))\n",
    "    ax[i].set_xlabel(\"Epochs\")\n",
    "    ax[i].set_ylabel(metric)\n",
    "    ax[i].legend([\"train\",\"val\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_scores=custommodel.evaluate(test_data,test_labels)\n",
    "print(\"Loss: \", test_scores[0])\n",
    "print(\"Accuracy: \", test_scores[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_labels=custommodel.predict(test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def roundoff(arr):\n",
    "    arr[np.argwhere(arr!=arr.max())]=0\n",
    "    arr[np.argwhere(arr==arr.max())]=1\n",
    "    return arr\n",
    "for labels in pred_labels:\n",
    "    labels=roundoff(labels)\n",
    "print(classification_report(test_labels,pred_labels,target_names=classes))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_ls=np.argmax(pred_labels,axis=1)\n",
    "test_ls=np.argmax(test_labels,axis=1)\n",
    "conf_arr=confusion_matrix(test_ls,pred_ls)\n",
    "plt.figure(figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')\n",
    "ax=sns.heatmap(conf_arr,cmap='Greens',annot=True,fmt='d',xticklabels=classes,yticklabels=classes)\n",
    "plt.title('Brain tumor Disease Diagonisis')\n",
    "plt.xlabel('Prediction')\n",
    "plt.ylabel('Truth')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Balanced Accuracy score {}%'.format(round(BAS(test_ls,pred_ls)*100,2)))\n",
    "print('Matthews Correleation Corrcoef {}%'.format(round(MCC(test_ls,pred_ls)*100,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "custom_incmodel_dir=wdr+\"BT_CNN_model6\"\n",
    "custommodel.save(custom_incmodel_dir,save_format='h5')\n",
    "os.listdir(wdr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "c=['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']\n",
    "ci=['glioma','meningioma','no-tumor','pituitary']\n",
    "# Load the image\n",
    "image = cv2.imread('./data2/test/meningioma/image(99).jpg')\n",
    "\n",
    "# Resize the image to match the expected input shape\n",
    "image = cv2.resize(image, (176, 176))\n",
    "\n",
    "# Normalize the image\n",
    "image = image / 255.0\n",
    "\n",
    "# Add an extra dimension to the image\n",
    "image = np.expand_dims(image, axis=0)\n",
    "\n",
    "# Load the model\n",
    "model = keras.models.load_model('./BT_CNN_model6')\n",
    "\n",
    "# Predict the class probabilities of the image\n",
    "prediction = model.predict(image)\n",
    "\n",
    "# Get the index of the class with the highest probability\n",
    "pci = np.argmax(prediction[0])\n",
    "\n",
    "# Display the result\n",
    "print(prediction[0])\n",
    "print('The predicted class is:', ci[pci])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import cv2\n",
    "# import numpy as np\n",
    "\n",
    "# # Load the image using cv2\n",
    "# image = cv2.imread('Te-pi_0116.jpg')\n",
    "\n",
    "# # Convert the image to a matrix array\n",
    "# image_array = np.array(image)\n",
    "\n",
    "# # Save the array to a file\n",
    "# np.save('image_array.npy', image_array)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import cv2\n",
    "# import numpy as np\n",
    "# import matplotlib.pyplot as plt\n",
    "\n",
    "# # Load the array from the .npy file\n",
    "# image_array = np.load('image_array.npy')\n",
    "\n",
    "# # Reconstruct the image using cv2\n",
    "# reconstructed_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "# # Display the reconstructed image\n",
    "# plt.imshow(reconstructed_image)\n",
    "# plt.axis('off')\n",
    "# plt.show()\n",
    "\n",
    "# # Alternatively, save the reconstructed image\n",
    "# cv2.imwrite('reconstructed_image.jpg', reconstructed_image);\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
