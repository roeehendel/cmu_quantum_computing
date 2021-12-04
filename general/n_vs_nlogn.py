import math


def f_40n(n):
    x = 0
    for i in range(n * 40):
        pass


def f_nlogn(n):
    x = 1
    logn = math.ceil(math.log2(n))
    for i in range(n * logn):
        pass


f_40n(10 ** 6)
f_nlogn(10 ** 6)
