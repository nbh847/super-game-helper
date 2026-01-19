from PIL import ImageGrab

import os

'''
获取截图
'''


def get_area_img(file_path_name, position_tuple):
    print('获取了file path: {} 的截图。'.format(file_path_name))
    im = ImageGrab.grab(position_tuple)
    im.save(file_path_name)


def delete_img(file_path_name):
    if os.path.exists(file_path_name):
        os.remove(file_path_name)
        print('已经删除: {}'.format(file_path_name))
    else:
        print('文件: {} 不存在'.format(file_path_name))

# file_path_name = "D:\\Python27\\workspace\\lushi_cheater\\screen_shut_file\\end_turn.jpg"
# position_tuple = (1065, 335, 1155, 363)
# get_area_img(file_path_name, position_tuple)
