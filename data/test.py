# def sortedSquares(nums):
#     left = 0
#     right = len(nums)-1
#     res = [0]*(right+1)
#     while left <= right:
#         l = abs(nums[left])
#         r = abs(nums[right])
#         if l < r:
#             print(right-left)
#             res[right-left] = r**2
#             right -= 1
#         else:
#             res[right-left] = l**2
#             left += 1
#     return res

# nums = [-4,-1,0,3,10]
# print(sortedSquares(nums))

# def rotate(nums, k: int):
#     """
#     Do not return anything, modify nums in-place instead.
#     """
#     n = len(nums)
#     k %= n
#     for i in range(k):
#         tmp = nums[-1]
#         for j in range(n-1,0,-1):
#             nums[j] = nums[j-1]
#         nums[0] = tmp
#     return nums

# nums = [1,2,3,4,5,6,7]
# k = 3
# print(rotate(nums,k))
# import math
# def isPalindrome(x):
#     if x < 0:
#         return False
#     elif x == 0:
#         return True
#     len_x = int(math.log10(x)) + 1
#     i = 0
#     half_nums = 0
#     temp = x
#     while i < len_x//2:
#         half_nums = int(half_nums*10 + temp%10)
#         temp /= 10
#         i += 1
#     if len_x%2==0:
#         x = int(x/(math.pow(10,i)))
#     else:
#         x = int(x/(math.pow(10,i+1)))
#     return int(half_nums) == int(x)

# print(isPalindrome(121))

import torch.nn as nn
import torch

x = torch.tensor([[-2, 1, 2, 6, 4], [-3, 1, 7, 2, -2], [-4, 2, 3, -1 , -3], [-7, 1, 2, 3, 11], [5, -7, 8, 12, -9]]).float()
print(x.size())
x = x.unsqueeze(0)
print(x.size())
y_1 = nn.MaxPool2d(kernel_size=2,stride=2, padding=0)
y_2 = nn.MaxPool2d(kernel_size=2,stride=2, padding=0, ceil_mode=True)
print(y_1(x))
print(y_2(x))