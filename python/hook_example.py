# encoding=utf-8

import datetime
import functools
import time

######
# 使用装饰器 decorator 进行 hook
# 参考: 廖旭峰博客 —— 在函数运行前后输出log : https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318435599930270c0381a3b44db991cd6d858064ac0000
# 参考: Stack Overflow —— 调用某函数func之前,先调用pre_func : https://stackoverflow.com/questions/35758323/hook-python-module-function

# ======= 自定义函数的时候,加上装饰器
# 不带参数的decorator
def log(func):
    def wrapper(*args, **kw):
        print("calling function: %s" % func.__name__)
        return func(*args, **kw)
    return wrapper

# 带参数的decorator
def log_2(text):
    def decorator_func(func):
        def wrapper(*args, **kw):
            print("显示log_2接受的参数: %s, 再执行函数: %s" % (text, func.__name__))
            result = func(*args, **kw)
            print("函数执行完毕")
            return result
        return wrapper
    return decorator_func

# 带参数的decorator,并且不会修改原函数的签名 __name__
def log_3(text):
    def decorator_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print("显示log_3接受的参数: %s, 再执行函数: %s" % (text, func.__name__))
            result = func(*args, **kw)
            print("函数执行完毕")
            return result
        return wrapper
    return decorator_func

def today():
    print(datetime.date.today())

@log
def today_use_log():
    print(datetime.date.today())

@log_2("'info_before_apply_func'")
def today_use_log_2():
    print(datetime.date.today())

@log_3("'info_before_apply_func'")
def today_use_log_3(a,b):
    print(datetime.date.today(),a,b)


print("=======> 直接调用函数 today :")
today()
print("\n=======> 函数定义前添加decorator :")
today_use_log()
print("\n=======> 函数定义前添加接受自定义参数的decorator :")
today_use_log_2()
print("函数签名是: %s" % today_use_log_2.__name__)
print("\n=======> 保持原函数签名不变,函数定义前添加接受自定义参数的decorator :")
today_use_log_3(1,10)
print("函数签名是: %s" % today_use_log_3.__name__)

# =============== hook 已有的函数
def wrap_func(ori_func,new_func):
    @functools.wraps(ori_func)
    def run(*args, **kwargs):
        return new_func(ori_func, *args, **kwargs)
    return run
#  1--- 随便某个函数
def replaced_func(ori_function, parameter):
    # 处理参数,如乘2倍
    new_params = parameter * 2
    return ori_function(new_params)
def target_func(cnt):
    return cnt+1
print("替换为replaced_func前的运行: target_func(2)=%s, 函数签名: %s" % (target_func(2), target_func.__name__))
target_func = wrap_func(target_func,replaced_func)
print("hook 之后: target_func(2)=%s, 函数签名: %s" % (target_func(2), target_func.__name__))
#   2--- hook print函数
def replaced_print(ori_function, parameter):
    now = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))
    new_params = now+": "+parameter
    return ori_function(new_params)
# old_print = print
print=wrap_func(print,replaced_print)
print("print方法已经被hook,会自动输出时间前缀")
