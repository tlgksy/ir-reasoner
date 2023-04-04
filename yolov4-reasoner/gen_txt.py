"""
Script that generates txt lists
"""

import os


def run():

    # get current path
    current_path = os.path.abspath(os.getcwd())
    train_img_path = current_path + os.sep + "coco" + os.sep + "images" + os.sep + "train2017" + os.sep
    val_img_path = current_path + os.sep + "coco" + os.sep + "images" + os.sep + "val2017" + os.sep

    # list the labels
    train_img_list = [x for x in os.listdir(train_img_path) if x.endswith('.jpg')]
    val_img_list = [x for x in os.listdir(val_img_path) if x.endswith('.jpg')]

    train_img_list = [train_img_path + x for x in train_img_list]
    val_img_list = [val_img_path + x for x in val_img_list]

    train_txt = current_path + os.sep + "coco" + os.sep + "train2017.txt"
    val_txt = current_path + os.sep + "coco" + os.sep + "val2017.txt"

    if not os.path.exists(train_txt):
        with open(train_txt, 'w') as f:
            f.write('\n'.join(train_img_list))
    else:
        print("Oops! Train txt already exists.")

    if not os.path.exists(val_txt):
        with open(val_txt, 'w') as f:
            f.write('\n'.join(val_img_list))
    else:
        print("Oops! Val txt already exists.")


if __name__ == '__main__':
    run()
