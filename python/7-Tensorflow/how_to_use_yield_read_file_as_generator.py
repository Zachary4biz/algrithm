# encoding=utf-8

################
# yield 必须在函数中使用,确定函数的返回值
# 可以在读文件时,实现类似f.readline()一行行取出,并且做一定封装(如每行都按\t分割,并转换为float等操作)
################



################
# 用于在循环中进行返回,但是每次都是只返回一次(类似于被中断了一样)
# 斐波那契数列, 例子参考: https://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/
################
def fibonacci():
    # yield之后,不需要写return,函数返回值直接是一个generator,每次用__next__() 获得下一个元素
    # 用循环遍历generator时会自动调用__next__()
    def fab_yield(cnt):
        n,a,b=0,0,1
        while n < cnt:
            yield b
            a, b = b, a+b
            n += 1

    # 通过返回数组的形式得到 斐波那契数列
    def fab_list(cnt):
        result = []
        n,a,b=0,0,1
        while n < cnt:
            result.append(b)
            a, b = b, a+b
            n += 1
        return result

    # 直接遍历生成斐波那契数列,并直接输出在shell中
    def fab_print(cnt):
        n,a,b=0,0,1
        while n < cnt:
            print (b)
            a, b = b, a+b
            n += 1

    print("yield 方式:")
    fab_yield(6) # 此时的输出应该为空
    generator = fab_yield(6)
    print("generator.__next__()", generator.__next__()) # 得到第一个数 "1"
    for i in generator: print("in the for loop", i)  # for循环中则是从第二个数 "1" 开始
    print("数组形式:", fab_list(6))
    print("print 方式:")
    fab_print(6) # 此时的输出为

# fibonacci()



def file_reader(path):
    # 每行都按句号分割,取第一句话
    def process_line(data):
        return data.split("。")[0]
    with open(path,"r") as f:
        for line in f:
            result = process_line(line)
            print ("***触发函数内部的print函数***")
            yield result

reader = file_reader("/Users/zac/result.txt")
for i in range(10): print(reader.__next__())

