# author: zac
# create-time: 2019/6/14 13:55
# usage: -

import itertools
import re
from collections import deque,defaultdict
import json

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
        self._init_from_file = {
            "json":self._parse_json,
            "csv":self._parse_csv
        }
        self.support_file_type = self._init_from_file.keys()


    def _parse_json(self,path:str):
        with open(path,"r") as f:
            res = json.load(f)
        self.init_from_dict(res)
        return res

    # \t分割理由和词,逗号做词间分割,如下
    # 0\tABC,BCD,ABC EFG
    def _parse_csv(self,path:str):
        res = {}
        with open(path,"r") as f:
            for reason,word_list_str in [line.strip().split("\t") for line in f]:
                res.update({reason:word_list_str.split(",")})
        self.init_from_dict(res)
        return res

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

    def init_from_file(self, file_path:str, file_type:str):
        return self._init_from_file[file_type](file_path)


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
        word_list = deque()
        for idx,char in enumerate(text):
            if char in self.root.children:
                j = idx
                p = self.root
                while j<len(text) and not p.is_leaf_node() and text[j] in p.children:
                    p = p.children[text[j]]
                    if p.is_end_word :
                        word_list.append(("".join(text[idx:j+1]),p.reason))
                        pass # 这里不用跳跃性地赋值 idx = idx+j,因为要兼容类似中文的语种,从 "我爱天安门" 中找到"我爱"和"爱天安门"
                    j += 1
        word_list = [(w.strip(), r) for w,r in word_list]
        word_list = [[k[0], k[1],len(list(g))] for k,g in itertools.groupby(sorted(word_list,key=None),key=None)] # IMPORTANT 注意groupby和sorted要使用相同的key,否则groupby可能失效
        # key不指定为词,按word和reason一起groupby;因为可能有一个词对应多个reason,如 Gaddfi 可能同时有政治敏感和宗教两个原因
        json_res = json.dumps([{"word":w, "reason":r, "cnt":c} for w,r,c in word_list])
        return json_res if return_json else word_list



if __name__ == '__main__':
    def test_one():
        dfa = DFATree()
        watch_list = {0:["violences", "violence","violence violences", "massacre", "porn",  "kill", "violate","violate purpose"]}
        print(">>> watch_list:",watch_list)
        dfa.init_from_dict(watch_list)
        content = "violence violences massacreadwgb violate, purpose, violate purpose, from now on massacre"
        print(">>> json_res: ",dfa.search(content))
        print(">>> res -h:")
        for i in dfa.search(content,return_json=False): print(i)
        dfa.search(content)

    def test_two():
        dfa = DFATree()
        dfa.init_from_file("/Users/zac/5-Algrithm/python/watch_list.json","json")
        content = "violence violences massacreadwgb violate, purpose, violate purpose, from now on massacre"
        print(">>> json_res: ",dfa.search(content))
        print(">>> res -h:")
        for i in dfa.search(content,return_json=False): print(i)
        dfa.search(content)

    test_one()
    test_two()

