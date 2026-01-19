# coding=utf-8
import math
import random
import time

import pyautogui

from lushi_cheater.logic.img_compare import compare
from lushi_cheater.logic.img_screen_grab import get_area_img, delete_img

''' 
工具库，提供各种集成方法
'''


class Utils:

    def __init__(self):
        pyautogui.FAILSAFE = True

    # 随机暂停1 - 3 s
    def pause_random_time(self):
        time.sleep(random.randint(1, 3))

    # 结束回合后移动到一个随机位置
    def move_to_random_place(self):
        pos_x = random.randint(318, 559)
        pos_y = random.randint(271, 436)
        self.mouse_move_by_humen_speed(pos_x, pos_y)

    # 定位鼠标的当前位置
    def locate_mouse_position(self):
        mouse_x, mouse_y = pyautogui.position()
        print('mouse_x, {}'.format(mouse_x))
        print('mouse_y, {}'.format(mouse_y))
        return mouse_x, mouse_y

    # 获取两点间的直线距离
    def get_length(self, p1, p2):
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        return math.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

    # 获取模拟人的速度去移动需要的时间
    def get_humen_time(self, pix_x, pix_y):
        # 鼠标目前的位置
        mouse_now = pyautogui.position()
        # 鼠标到目标的距离
        mouse_locate_length = self.get_length(mouse_now, (pix_x, pix_y))
        # 人的鼠标移动速度
        humen_speed = random.randint(750, 880)
        # 移动花费的总时间
        whole_time = mouse_locate_length / humen_speed
        return whole_time

    # 模拟人的速度去移动
    def mouse_move_by_humen_speed(self, pix_x, pix_y):
        whole_time = self.get_humen_time(pix_x, pix_y)
        pyautogui.moveTo(pix_x, pix_y, whole_time)

    # 模拟人的速度去拖拽
    def mouse_drog_by_humen_speed(self, pix_x, pix_y):
        whole_time = self.get_humen_time(pix_x, pix_y)
        pyautogui.dragTo(pix_x, pix_y, whole_time)

    # 结束我的回合
    def end_my_turn(self):
        mouse_x = random.randint(1080, 1148)
        mouse_y = random.randint(340, 358)
        self.mouse_move_by_humen_speed(mouse_x, mouse_y)
        pyautogui.click()
        print('已结束我的回合')
        self.move_to_random_place()

    # 使用英雄技能
    def use_hero_skill(self):
        mouse_x = random.randint(789, 835)
        mouse_y = random.randint(565, 604)
        self.mouse_move_by_humen_speed(mouse_x, mouse_y)
        pyautogui.click()
        print('已使用英雄技能')

    # 使用武器攻击敌方英雄
    def use_knife(self):
        start_x = random.randint(650, 720)
        start_y = random.randint(580, 628)
        end_x = random.randint(651, 721)
        end_y = random.randint(126, 173)
        whole_time = self.get_humen_time(end_x, end_y)
        self.mouse_move_by_humen_speed(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, 1)
        pyautogui.rightClick()

    # 判断是否是我的回合，如果有结束按钮，代表是我的回合
    def is_my_turn(self):
        position = (1065, 335, 1155, 363)
        store_path = "D:\\Python27\\workspace\\lushi_cheater\\screen_shut_file\\screen_end_btn.jpg"
        now_btn_img = get_area_img(store_path, position)
        compare_img_path_1 = "D:\\Python27\\workspace\\lushi_cheater\\imgs\\first_turn_end_yellow.jpg"
        compare_img_path_2 = "D:\\Python27\\workspace\\lushi_cheater\\imgs\\first_turn_end_green.jpg"
        compare1 = compare(store_path, compare_img_path_1)
        compare2 = compare(store_path, compare_img_path_2)
        # 删除临时截图
        delete_img(store_path)
        if compare1 <= 25 or compare2 <= 19:
            print('是我的回合')
            return True
        else:
            print('不是我的回合')
            return False

    # 把随从从手里拖到场上
    def drag_cards_to_ground(self):
        pix_x = random.randint(316, 1029)
        pix_y = random.randint(390, 467)
        whole_time = self.get_humen_time(pix_x, pix_y)
        pyautogui.dragTo(pix_x, pix_y, whole_time)

    # 让随从打敌方英雄的脸
    def drag_cards_to_fight_hero(self):
        pix_x = random.randint(652, 725)
        pix_y = random.randint(127, 172)
        whole_time = self.get_humen_time(pix_x, pix_y)
        pyautogui.dragTo(pix_x, pix_y, whole_time)
        pyautogui.rightClick()

    # 让能行动的随从攻击敌方英雄
    def follower_action(self):
        index = 0
        for i in range(10):
            print('开始打敌方英雄的脸第{}次'.format(i + 1))
            start_x = 1013 + random.randint(-10, 10) + index
            start_y = 429 + random.randint(-29, 29)
            self.mouse_move_by_humen_speed(start_x, start_y)
            self.drag_cards_to_fight_hero()
            index -= 68

    # 移动到底部最左边打出所有能打出的随从
    def move_cards(self):
        index = 0
        for i in range(10):
            print('开始移动第{}次'.format(i + 1))
            start_x = 453 + random.randint(-10, 10) + index
            start_y = 734 + random.randint(-29, 29)
            self.mouse_move_by_humen_speed(start_x, start_y)
            self.drag_cards_to_ground()
            index += 47
        # 把场上所有能行动的随从全部行动
        self.follower_action()

    # 确认是否要点击游戏确认开始按钮
    def confirm_game_start(self):
        position = (660, 597, 716, 619)
        store_path = "D:\\Python27\\workspace\\lushi_cheater\\screen_shut_file\\game_confirm_start.jpg"
        get_area_img(store_path, position)
        compare_img_path = "D:\\Python27\\workspace\\lushi_cheater\\imgs\\game_confirm_start.jpg"
        compare_result = compare(store_path, compare_img_path)
        # 删除临时截图
        delete_img(store_path)
        if compare_result <= 7:
            print('需要点击游戏确认按钮')
            mouse_x = random.randint(660, 716)
            mouse_y = random.randint(597, 619)
            self.mouse_move_by_humen_speed(mouse_x, mouse_y)
            pyautogui.click()
            print('已点击')
        else:
            print('不需要点击游戏确认按钮')

    # 确认是否需要点击标准对战按钮
    def confirm_standard_game_begin(self):
        position = (963, 591, 1036, 662)
        store_path = "D:\\Python27\\workspace\\lushi_cheater\\screen_shut_file\\game_standard_begin.jpg"
        get_area_img(store_path, position)
        compare_img_path = "D:\\Python27\\workspace\\lushi_cheater\\imgs\\game_standard_begin.jpg"
        compare_result = compare(store_path, compare_img_path)
        # 删除临时截图
        delete_img(store_path)
        if compare_result <= 12:
            print('需要点击标准对战按钮')
            mouse_x = random.randint(963, 1036)
            mouse_y = random.randint(591, 662)
            self.mouse_move_by_humen_speed(mouse_x, mouse_y)
            pyautogui.click()
            print('已点击')
        else:
            print('不需要点击标准对战按钮')

    # 使用不同的英雄的攻击方式
    def attack_by_hero(self, hero='战士'):
        if hero == '战士' or hero == '圣骑士' or hero == '潜行者':
            # 圣骑士，战士的攻击方式
            # 使用武器攻击敌方英雄
            self.use_knife()
            # 使用英雄技能
            self.use_hero_skill()
        if hero == '德鲁伊':
            # 德鲁伊的攻击方式
            # 先使用英雄技能，再攻击敌方英雄
            self.use_hero_skill()
            self.use_knife()
        if hero == '法师':
            # 法师的攻击方式
            ...
        if hero == '术士':
            # 术士的攻击方式
            ...
        if hero == '牧师':
            # 牧师的攻击方式
            mouse_x = random.randint(789, 835)
            mouse_y = random.randint(565, 604)
            self.mouse_move_by_humen_speed(mouse_x, mouse_y)
            mouse_x = random.randint(649, 711)
            mouse_y = random.randint(573, 611)
            self.mouse_drog_by_humen_speed(mouse_x, mouse_y)
            pyautogui.rightClick()

    # 游戏行动模块
    def play_game_card(self):
        while True:
            # 点击一下，确认可以退出每局结束的比赛
            pyautogui.click(683, 458)
            self.mouse_move_by_humen_speed(862, 423)
            pyautogui.click()
            self.mouse_move_by_humen_speed(683, 458)
            # 检查是否需要点击标准对战
            self.confirm_standard_game_begin()
            # 检查是否需要点击确认游戏
            self.confirm_game_start()
            # time.sleep(2000)

            # 检查是否是本回合
            if self.is_my_turn():
                # 暂停6s等待所有的卡牌加载完
                time.sleep(random.randint(3, 6))
                # 开始出牌
                self.move_cards()
                # 使用不同的英雄的攻击方式
                # self.attack_by_hero('战士')
                # self.attack_by_hero('德鲁伊')
                # self.attack_by_hero('法师')
                # self.attack_by_hero('术士')
                self.attack_by_hero('牧师')
                # 结束本回合，之后检查是否到了自己行动的回合
                self.end_my_turn()
            else:
                print('每5s检查一次是否到了我的回合')
                time.sleep(random.randint(2, 4))

    # 游戏运行入口
    def run(self):
        # 定位鼠标位置
        self.locate_mouse_position()
        time.sleep(10)
        # 点击图标进入游戏主界面
        self.mouse_move_by_humen_speed(210, 741)
        pyautogui.click()
        time.sleep(random.randint(1, 3))
        # 点击最下方的牌库循环出牌
        self.play_game_card()


if __name__ == '__main__':
    u = Utils()
    u.run()
