# author: zac
# create-time: 2019/6/14 10:52
# usage: -
from itertools import takewhile,islice
import sys
print(sys.version)
from collections import deque
import itertools
import re
import json

# class Node(object):
#     def __init__(self):
#         self.children = None # None marked as last-char(node) of a word
#         self.reason = None # None marked as in the middle of a word. last-char(node) will add a reason
#
# def add_word(root:Node_, word:str, reason:int=0):
#     node = root
#     for idx,voc in enumerate(word):
#         if node.children is None:
#             node.children = {word[idx]:Node_()}
#         elif word[idx] not in node.children:
#             node.children[word[idx]] = Node_()
#         node = node.children[word[idx]]
#     node.reason=reason
#
# def add_word(root, word,reason):
#     node = root
#     for idx, char in enumerate(word):
#         node.add_child(char)
#         node = node.children[char]
#     node.reason = reason

# def init(watch_list=None):
#     if watch_list is None:
#         watch_list = {0:["massacre", "porn", "violence", "kill", "violate"]}
#     root_ = Node()
#     for reason,voc_list in watch_list.items():
#         for voc in voc_list:
#             add_word(root_," "+voc+" ")
#     return root_
#
# def is_contain(message_inp, root):
#     message = " " + message_inp +" "
#     for i in range(len(message)):
#         p = root
#         print("\nroot-p: ",p)
#         print("root-p.children: ",p.children)
#         print("root-p.reason: ",p.reason)
#         j = i
#         print("message[i] ", message[i])
#         while j < len(message) and p.children is not None and message[j] in p.children:
#             print(" ")
#             print("  message[j] ",message[j])
#             p = p.children[message[j]]
#             print("  child-p: ",p)
#             print("  child-p.children: ",p.children)
#             print("  child-p.reason: ",p.reason)
#             j += 1
#         if p.children is None:
#             return True
#     return False

# root = init()
# print("root is:", root)
# print("root.children is:", root.children)
# print(is_contain("massacreadwgb violate",root))

class Node(object):
    """
    DFA不需要反查,所以未记录父节点
    在DFA中叶子节点标记了一个词的结束如, "porn "和"porno"中的" "和"o" |
      - 这种用叶子节点直接标记词汇结束的做法,不兼容 "既要屏蔽A也要屏蔽AB",但其实从业务逻辑上来说,写"A"就不用"AB",写"AB"表示要细分的那就不需要"A"
      - 例如: " porn "和" porn abc "会把" porn "的最后的空格加上children={"a":xxx}
      - 其实也不支持 "vocab" 和 "vocabulary"
    """
    def __init__(self, reason=None):
        self.children = None
        self.reason = reason # 仅end_word==True时有值, 标记某个词是什么原因被标记为敏感词
        self.is_end_word = False # 标记词条结束(不再用叶子节点作为结束标志)

    def is_leaf_node(self):
        return self.children is None

    def add_child(self,v):
        if self.is_leaf_node():
            self.children = {v:Node()}
        elif v not in self.children:
            self.children[v] = Node()
        else:
            pass # DFA子节点已有则不更新


class DFATree(object):
    def __init__(self):
        self.root = Node()

    def add_word(self,word_inp,reason,sep = " "):
        word = sep+word_inp+sep
        node = self.root
        for idx, char in enumerate(word):
            node.add_child(char)
            node = node.children[char]
        node.reason = reason
        node.is_end_word = True

    def add_word_CN(self,word_inp,reason):
        self.add_word(word_inp,reason,sep="")

    def add_word_EN(self,word_inp,reason):
        self.add_word(word_inp,reason,sep=" ")

    def init_from_dict(self,watch_list_inp:dict):
        for reason,word_list in watch_list_inp.items():
            for word in word_list:
                self.add_word_EN(word,reason)

    # dfa.getRes(" violence") 获得字符串最后一个字母"e"的Node
    def node_of_last_char(self,ss) -> Node:
        bn = self.root
        for idx,c in enumerate(ss):
            bn = bn.children[c]
        return bn

    # 清理句子中的各种符号
    @staticmethod
    def text_format(text_inp:str):
        return " " + re.sub("[^\\w]+"," ",text_inp) + " "


    # do_fromat控制是否自动清理符号
    def search(self, text_inp:str, return_json:bool=True, do_format:bool=True):
        text = self.text_format(text_inp) if do_format else text_inp
        print(text)
        word_list = deque()
        for idx,char in enumerate(text):
            p = self.root
            # print("\nroot-p: ",p)
            print("root-p.children: ",p.children)
            print("root-p.reason: ",p.reason)
            if char in self.root.children:
                j = idx
                p = self.root
                while j<len(text) and text[j] in p.children:
                    p = p.children[text[j]]
                    print(text[j],p.children)
                    if p.is_end_word :
                        word_list.append(("".join(text[idx:j+1]),p.reason))
                        break # 这里不用跳跃性地赋值 idx = idx+j,因为要兼容类似中文的语种,从 "我爱天安门" 中找到"我爱"和"爱天安门"
                    j += 1

        word_list = [(w.strip(), r) for w,r in word_list]
        word_list = [[k[0], k[1],len(list(g))] for k,g in itertools.groupby(word_list)]
        # key不指定为词,按word和reason一起groupby;因为可能有一个词对应多个reason,如 Gaddfi 可能同时有政治敏感和宗教两个原因
        json_res = json.dumps([{"word":w, "reason":r, "cnt":c} for w,r,c in word_list])
        return json_res if return_json else word_list

dfa = DFATree()

# text_inp = "massacreadwgb violate"
content = "violence violences massacreadwgb violate, purpose, violate purpose, from now on massacre"
watch_list = {0:["violences", "violence","violence violences", "massacre", "porn",  "kill", "violate","violate purpose"]}
dfa.init_from_dict(watch_list)
# print(">>>",dfa.search("massacreadwgb violate"))
print(">>>清理符号:", dfa.search(content))
word_list = dfa.search(content,return_json=False)
print(">>> json",json.dumps([{"word":k[0], "reason":k[1], "cnt":len(list(g))} for k,g in itertools.groupby(word_list)]))


# print(">>>不清理符号:", dfa.search(content, do_format=False))

test_text = " " + re.sub("[^\\w]+"," ",content) + " "
print("'"+test_text[0:11]+"'")
print(test_text[8],test_text[9],test_text[10])
basic_node = dfa.root
print("basic_node: ",basic_node)
for idx,c in enumerate(" violence "):
    basic_node = basic_node.children[c]
    print(idx,":","'"+c+"'",basic_node,basic_node.children)
    if idx in [0,8,9]:
        print("basic_node of %s" % ("'"+c+"'"),basic_node)

print(">>>>>>> checking")
for i in [" "," violence "," vio"," violence violences "]:
    bn = dfa.node_of_last_char(i)
    print(i,"[end_word]:",bn.is_end_word,"[leaf_node]:",bn.is_leaf_node(),"[reason]:",bn.reason)

print("children at ' violence':",dfa.node_of_last_char(" violence").children)
print("children at ' violence ':",dfa.node_of_last_char(" violence ").children)
print("children at ' kill':",dfa.node_of_last_char(" kill").children)
print("children at ' kill ':",dfa.node_of_last_char(" kill ").children)



