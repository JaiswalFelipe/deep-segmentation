{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "mydataloader.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "mount_file_id": "1Ayc38Qo9yOJM72pp5-7GJHKBmwilKuTm",
      "authorship_tag": "ABX9TyPiWiBNYblltY2vbgjjfgaY",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JaiswalFelipe/deep-segmentation/blob/master/mydataloader.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import numpy as np\n",
        "import imageio\n",
        "import torch\n",
        "\n",
        "from PIL import Image\n",
        "from torch.utils.data import Dataset\n",
        "from skimage import transform\n",
        "from skimage import img_as_float\n",
        "from matplotlib import pyplot as plt\n",
        "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n",
        "from torchvision import transforms\n",
        "\n",
        "#from data_utils import create_or_load_statistics, create_distrib, normalize_images, data_augmentation\n",
        "\n",
        "scaler = MinMaxScaler()"
      ],
      "metadata": {
        "id": "EG8HkKPNQ1iV"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def compute_image_mean(data):\n",
        "    _mean = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)\n",
        "    _std = np.std(np.std(np.std(data, axis=0, ddof=1), axis=0, ddof=1), axis=0, ddof=1)\n",
        "\n",
        "    return _mean, _std\n",
        "\n",
        "\n",
        "def normalize_images(data, _mean, _std):\n",
        "    for i in range(len(_mean)):\n",
        "        data[:, :, i] = np.subtract(data[:, :, i], _mean[i])\n",
        "        data[:, :, i] = np.divide(data[:, :, i], _std[i])\n",
        "\n",
        "\n",
        "\n",
        "def data_augmentation(img, label=None):\n",
        "    rand_fliplr = np.random.random() > 0.50\n",
        "    rand_flipud = np.random.random() > 0.50\n",
        "    rand_rotate = np.random.random()\n",
        "\n",
        "    if rand_fliplr:\n",
        "        img = np.fliplr(img)\n",
        "        label = np.fliplr(label)\n",
        "    if rand_flipud:\n",
        "        img = np.flipud(img)\n",
        "        label = np.flipud(label)\n",
        "\n",
        "    if rand_rotate < 0.25:\n",
        "        img = transform.rotate(img, 270, order=1, preserve_range=True)\n",
        "        label = transform.rotate(label, 270, order=0, preserve_range=True)\n",
        "    elif rand_rotate < 0.50:\n",
        "        img = transform.rotate(img, 180, order=1, preserve_range=True)\n",
        "        label = transform.rotate(label, 180, order=0, preserve_range=True)\n",
        "    elif rand_rotate < 0.75:\n",
        "        img = transform.rotate(img, 90, order=1, preserve_range=True)\n",
        "        label = transform.rotate(label, 90, order=0, preserve_range=True)\n",
        "\n",
        "    img = img.astype(np.float32)\n",
        "    label = label.astype(np.int64)\n",
        "\n",
        "    return img, label\n"
      ],
      "metadata": {
        "id": "pmVrwGdJucs1"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class NGDataset(Dataset):\n",
        "  def __init__(self, img_dir, mask_dir, img_list, mean, std, transform=None, if_test = False):\n",
        "\n",
        "    self.img_dir = img_dir\n",
        "    self.mask_dir = mask_dir\n",
        "    self.img_list = img_list\n",
        "    self.mean = mean\n",
        "    self.std = std\n",
        "    self.transform = transform\n",
        "    self.if_test = if_test\n",
        "    #self.images = os.listdir(img_dir)\n",
        "\n",
        "  def __len__(self):\n",
        "    return len(self.img_list)\n",
        "\n",
        "  def __getitem__(self, index):\n",
        "    \n",
        "    image = img_as_float(imageio.imread(os.path.join(self.img_dir, self.img_list[index] + '.tif')))\n",
        "    mask = imageio.imread(os.path.join(self.mask_dir, self.img_list[index] + '.tif')).astype(int)\n",
        "\n",
        "    if self.transform is not None:\n",
        "      transformed_img = self.transform(image = image, mask = mask)\n",
        "\n",
        "      image = transformed_img['image']\n",
        "      mask = transformed_img['mask']\n",
        "\n",
        "    #transformed_images = transform.Compose([transform.ToTensor(), transform.Normalize(self.mean, self.std)])\n",
        "\n",
        "    # Normalization.\n",
        "    normalize_images(image, self.mean, self.std) # check data_utils.py\n",
        "    \n",
        "    if self.if_test:\n",
        "      pass \n",
        "    else: \n",
        "      image, mask = data_augmentation(image, mask)\n",
        "     \n",
        "    image = np.transpose(image, (2, 0, 1))\n",
        "\n",
        "    # Turning to tensors.\n",
        "    image = torch.from_numpy(image.copy())\n",
        "    mask = torch.from_numpy(mask.copy())\n",
        "\n",
        "    # Returning to iterator.\n",
        "    return image.float(), mask\n"
      ],
      "metadata": {
        "id": "Q7UNPeWWDVF3"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "img_path = '/content/drive/MyDrive/training_patches/image_dataset'\n",
        "mask_path = '/content/drive/MyDrive/training_patches/mask_dataset'\n",
        "\n",
        "img_list = []\n",
        "for (_,_,files) in os.walk(img_path):\n",
        "     for file in files:   \n",
        "        img_list.append(file.split('.')[0])\n",
        "        #print (files)\n",
        "        #print ('--------------------------------')"
      ],
      "metadata": {
        "id": "JoBevDAxn688"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_loader = NGDataset(img_path, mask_path, img_list, mean, std)"
      ],
      "metadata": {
        "id": "x-R2MNNomgIM"
      },
      "execution_count": 8,
      "outputs": []
    }
  ]
}