import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Env state 
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


# # simple actions_dim = 7 
# SIMPLE_MOVEMENT = [
#     ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#     ["A"],          # Jump straight up.
#     ["left"],       # Move left.
# ]
#-----------------------------------------------------------------------------
#獎勵函數
'''
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

'''
'''
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
'''

#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)

# 鼓勵跳躍和滯空
def distance_y_offset_reward(info, reward, prev_info):
    total_reward = reward
    
    curr_y = info.get('y_pos', 0)
    prev_y = prev_info.get('y_pos', 0)
    
    jump_height = curr_y - prev_y  # 計算跳躍高度
    is_jumping = jump_height > 0
    
    if is_jumping:
        if jump_height > 15:  # 高跳（>15）
            total_reward += 100  # 獎勵高跳
        elif jump_height > 10:  # 中跳（10~15）
            total_reward += 50
        elif jump_height > 5:  # 小跳（5~10）
            total_reward += 10
    
    return total_reward

# 鼓勵前進，懲罰後退或停留
def distance_x_offset_reward(info, reward, prev_info):
    total_reward = reward
    
    curr_x = info.get('x_pos', 0)
    prev_x = prev_info.get('x_pos', 0)
    if curr_x > prev_x:
        total_reward += 30
    elif curr_x < prev_x:
        total_reward -= 20
    
    return total_reward

# 用來獎勳玩家蒐集硬幣的行為
def get_coin_reward(info, reward, prev_info):
    total_reward = reward
    
    current_coins = info.get('coins', 0)
    previous_coins = prev_info.get('coins', 0)
    if current_coins > previous_coins:
        total_reward += (current_coins - previous_coins) * 5
    
    return total_reward

# 鼓勵擊敗怪物
def monster_score_reward(info, reward, prev_info):
    total_reward = reward
    
    current_killed = info.get('score', 0)
    previous_killed = prev_info.get('score', 0)
    if current_killed > previous_killed:
        total_reward += (current_killed - previous_killed) * 10
    
    return total_reward

# 鼓勵完成關卡（終點旗幟）
def final_flag_reward(info, reward):
    total_reward = reward
    
    flag_status = info.get('flag', 0)
    if flag_status == 1:
        total_reward += 100000
    
    return total_reward

# 懲罰原地重複跳躍
# 避免玩家在水管或障礙物前重複跳躍

def punish_repeated_jump(info, reward, prev_info):
    
    curr_y = info.get('y_pos', 0)
    prev_y = prev_info.get('y_pos', 0)
    curr_x = info.get('x_pos', 0)
    prev_x = prev_info.get('x_pos', 0)
    
    jump_height = curr_y - prev_y
    is_repeated_short_jump = (curr_x == prev_x) and (0 < jump_height <= 5)
    
    if is_repeated_short_jump:
        reward -= 10  # 懲罰重複短跳
    
    return reward

def punish_falling_into_hole_at_epoch_end(info, reward, prev_info):
    """
    在每個 epoch 結束時檢查是否有掉洞或死亡情況。
    """
    total_reward = reward

    # 獲取最後一個狀態和當前狀態的生命值
    prev_life = prev_info.get('life', 0)
    curr_life = info.get('life', 0)

    # 如果生命值減少，表示死亡事件
    if curr_life < prev_life:
        total_reward -= 1000  # 掉洞懲罰分數

    return total_reward

#===============to do==========================================