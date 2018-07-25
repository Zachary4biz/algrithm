import argparse

parser = argparse.ArgumentParser(description=" desription 显示在help信息之前", epilog=" epilog 显示在help信息之后", usage="此脚本的目的是更新文件")
parser.add_argument("--argument1", type=int, help='argument1 类型为整数')
parser.add_argument("--argument2", choices=['1','2'], help='argument2 只能是 1或2,类型为字符串')
parser.add_argument("--argument3", type=int, choices=[1, 2], help='argument3 只能是 1或2 类型是数字')
parser.add_argument("--argument4", type=str, default="defualt_arg4", help='a help for argument4')
print(parser.parse_args())
args = parser.parse_args()
print("arg1 is "+ str(args.argument1))
print("arg2 is "+ str(args.argument2))
print("arg3 is "+ str(args.argument3))
print("arg4 is "+ str(args.argument4))
