# encoding:utf-8
# author: zac
# create-time: 2019/6/14 13:55
# usage: - python2.7 checked(✅)

from __future__ import print_function
import itertools
import re
from collections import deque
import json
import os
import sys
from io import open
import collections

class Node(object):
    """
    DFA不需要反查,所以未记录父节点
    在DFA中叶子节点标记了一个词的结束如, "porn "和"porno"中的" "和"o" |
      - 这种用叶子节点直接标记词汇结束的做法,不兼容 "既要屏蔽A也要屏蔽AB",但其实从业务逻辑上来说,写"A"就不用"AB",写"AB"表示要细分的那就不需要"A"
      - 例如: " porn "和" porn abc "会把" porn "的最后的空格加上children={"a":xxx}
      - 其实也不支持 "vocab" 和 "vocabulary" # 经验证是支持的
    """
    def __init__(self, reason=None):
        self.children = None
        self.reason = reason # 仅end_word==True时有值, 标记某个词是什么原因被标记为敏感词
        self.is_end_word = False # 标记词条结束(不再用叶子节点作为结束标志)

    def is_leaf_node(self):
        return self.children is None

    def children_names(self):
        return self.children.keys() if self.children is not None else None

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


    def _parse_json(self,path):
        with open(path,"r") as f:
            res = json.load(f)
        self.init_from_dict(res)
        return res

    # \t分割理由和词,逗号做词间分割,如下
    # 0\tABC,BCD,ABC EFG
    def _parse_csv(self,path):
        res = {}
        with open(path,"r") as f:
            for reason,word_list_str in [line.strip().split("\t") for line in f]:
                res.update({reason:word_list_str.split(",")})
        self.init_from_dict(res)
        return res

    def _add_word(self, word_inp, reason, sep =" "):
        word = sep+word_inp+sep
        node = self.root
        for idx, char in enumerate(word):
            node.add_child(char)
            node = node.children[char]
        node.reason = reason
        node.is_end_word = True

    def add_word_CN(self,word_inp,reason):
        self._add_word(word_inp, reason, sep="")

    def add_word_EN(self,word_inp,reason):
        self._add_word(word_inp, reason, sep=" ")

    def init_from_file(self, file_path, file_type):
        return self._init_from_file[file_type](file_path)


    def init_from_dict(self,watch_list_inp):
        for reason,word_list in watch_list_inp.items():
            for word in word_list:
                self.add_word_EN(word,reason)

    # dfa.getRes(" violence") 获得字符串最后一个字母"e"的Node
    def node_of_last_char(self,ss):
        bn = self.root
        for idx,c in enumerate(ss):
            bn = bn.children[c]
        return bn

    # 清理句子中的各种符号
    @staticmethod
    def text_format(text_inp):
        emoji_regex = """|(\ud83d[\ude00-\ude4f])"""
        symbol_regex = """([~!@#$%^&*()_+-\={}|\[\]\\\\:";'<>?,./])""" # 印地语或者其他亚洲语言都不属于 \w+ 所以不能这样去除符号
        regex = symbol_regex + emoji_regex
        clean_symobl = re.sub(regex," ",text_inp)
        clean_whitespace = re.sub("\\s+"," "," " +clean_symobl+ " ")
        return  clean_whitespace

    # 直接返回groupby的结果，即[(1, <itertools._grouper at 0x10d4ae610>), (2, <itertools._grouper at 0x10d4ae550>)]
    # 注意带itertools._grouper的实际结果是 [(1, [('a', 1), ('b', 1)]), (2, [('c', 2), ('d', 2)])]
    # 需要取出_grouper里每项的首项
    def _get_all_children_asIter(self,verbose=False):
        res_list = []
        def loop_to_show(cur_node, cur_res_list, cur_word, level=1):
            if cur_node.is_end_word:
                if not cur_node.is_leaf_node():
                    # 是end_word却不是leaf_node，说明是chunk词中间那个空格，例如" cunt " 和 " cunt face "，
                    # 1. 此时的end_word先加入到结果中
                    cur_res_list.append((cur_word, cur_node.reason))
                    # 2. 继续向下查找
                    for key_, sub_node in cur_node.children.items():
                        if verbose: print("|" * level + " " + key_ + "\t_" + cur_word)
                        loop_to_show(sub_node, cur_res_list, cur_word + key_, level + 1)
                else:
                    # 既是end_word又是leaf_node，就是一个普通的single word，直接append
                    cur_res_list.append((cur_word, cur_node.reason))
                if verbose: print(cur_res_list[-1])
            else:
                for key_, sub_node in cur_node.children.items():
                    if verbose: print("|" * level + " " + key_ + "\t_" + cur_word)
                    loop_to_show(sub_node, cur_res_list, cur_word + key_, level+1)
        loop_to_show(self.root,res_list,cur_word = "")
        the_key = lambda x: x[1]
        res_asIter = itertools.groupby(sorted(res_list, key=the_key), key=the_key)
        return res_asIter

    # 解开groupby的iter.grouper，即取出_grouper里每项的首项，得到例如：[(1, ['a', 'b']), (2, ['c', 'd'])]
    def get_all_children_asList(self,verbose=False):
        res_asIter = self._get_all_children_asIter(verbose)
        res_asDict = dict([(k, map(lambda x: x[0][1:-1], g)) for k, g in res_asIter])
        return res_asDict

    # do_fromat控制是否自动清理符号
    def search(self, text_inp, return_json=True, do_format=True):
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
    def test_init_from_dict(content ="   violence violences massacreadwgb violate, purpose, violate purpose, from now on massacre"):
        print(">>> AT TEST-ONE:")
        dfa = DFATree()
        watch_list = {0:["violences", "violence","violence violences", "massacre", "porn",  "kill", "violate","violate purpose"]}
        print(">>> watch_list:",watch_list)
        dfa.init_from_dict(watch_list)
        print(">>> json_res: ",dfa.search(content))
        print(">>> res -h:")
        for i in dfa.search(content,return_json=False): print(i)
        dfa.search(content)

    def test_init_from_json_file(content ="बिस्तर गर्म, कामक्रीड़ा, सेक्सी"):
        print("\n\n>>> AT TEST-TWO:")
        dfa = DFATree()
        dfa.init_from_file(os.path.dirname(__file__)+"/TestCase/watch_list.json","json")
        print(">>> json_res: ",dfa.search(content))
        print(">>> res -h:")
        for i in dfa.search(content,return_json=False): print(i)
        dfa.search(content)

    def test_format_hindi():
        print("印地语看着是一个字实际上是多个字符:")
        hindi = "बिस्तर गर्म वीर्य हिलती कार"
        print(len(hindi))
        for i in hindi:
            print(i)
            print(i in hindi)
        print(hindi)
        print(DFATree.text_format(hindi))
        print("印地语的 re.sub: ")
        content = " बिस्तर गर्म , कामक्रीड़ा , सेक्सी  ~सेक्सी!सेक्सी@सेक्सी#सेक्सी$सेक्सी%सेक्सी^सेक्सी&सेक्सी*सेक्सी(सेक्सी)सेक्सी_सेक्सी+सेक्सी-सेक्सी=सेक्सी{सेक्सी}सेक्सी|सेक्सी[सेक्सी]सेक्सी\सेक्सी:सेक्सी\"सेक्सी;सेक्सी'सेक्सी<सेक्सी>सेक्सी?सेक्सी,सेक्सी.सेक्सी/सेक्सी'सेक्सी"
        print(content)
        print("re.sub2",re.sub("""[~!@#$%^&*()_+-={}|\[\]\\\\:";'<>?,./]"""," \u2717 ",content))

    def test_case_from_file():
        print("\n\n>>> AT TEST_FROM_FILE")
        dfa = DFATree()
        dfa.init_from_file(os.path.dirname(__file__)+"/TestCase/watch_list.json","json")
        with open(os.path.dirname(__file__)+"/TestCase/hindi_test_case.txt","r+") as f:
            test_case = [i.strip() for i in f.readlines()]
        for w in test_case :
            print("test-case: ",w)
            print("text_formatted: ", DFATree.text_format(w))
            print("result: ", dfa.search(w,return_json=False))

    def test_show_all_words():
        dfa = DFATree()
        dfa.init_from_file(os.path.dirname(__file__)+"/TestCase/watch_list.json","json")
        dfa_wordsDict = dfa.get_all_children_asList(verbose=False)
        print(">>> all_words:")
        for k,v in dfa_wordsDict.items():
            print(k+"\t"+", ".join(v))

    def test_on_get_all_children_asList(verbose=False):
        print("\n\n[AT TEST_ON_GET_ALL_CHILDREN_ASLIST]...")
        dfa = DFATree()
        dfa.init_from_file(os.path.dirname(__file__) + "/TestCase/watch_list.json", "json")
        dfa_wordsDict = dfa.get_all_children_asList(verbose=False)
        with open(os.path.dirname(__file__)+"/TestCase/watch_list.json","r",encoding="utf-8") as f:
            dic = json.load(f)
            for key,item in dic.items():
                item = list(set(item))
                success = sorted(item) == sorted(dfa_wordsDict[key])
                print(">>> check at [key - 是否完全相同]: [{} - {}]".format(key, str(success)))
                if verbose and not success:
                    print("[file & dfa]:\t"+", ".join(set(item).intersection(dfa_wordsDict[key])))
                    print("[file - dfa]:\t"+", ".join(set(item).difference(dfa_wordsDict[key])))
                    print("[dfa - file]:\t" + ", ".join(set(dfa_wordsDict[key]).difference(item)))

    def test_check_wordChunk():
        """
        如下所示，如果加入的是chunk的词组("cunt","cunt face")，会在cunt的词尾出现标记为is_end_word=True，但is_leaf_node()=False
        其他情况下（单个词）的is_end_word和is_leaf_node()都是同步的
        """
        print("\n\n[AT TEST_CHECK_WORDCHUNK]...")
        dfa = DFATree()
        dfa.add_word_EN("cunt","1")
        dfa.add_word_EN("cunt face","2")
        dfa.add_word_EN("bitch","1")
        dfa.add_word_EN("नग्न","1")
        dfa.add_word_EN("नग्न तस्वीरें","1")
        for k,v in dfa.get_all_children_asList().items():
            print(k+"\t"+", ".join(v))
        def log_word(word):
            print("%s 词尾（最后一个空格）的children有哪些: " % word, dfa.node_of_last_char(" "+word+" ").children_names())
            print("%s 词尾（最后一个空格）是否为叶子节点?" % word,dfa.node_of_last_char(" "+word+" ").is_leaf_node())
            print("%s 词尾（最后一个空格）是否为end_word?" % word, dfa.node_of_last_char(" "+word+" ").is_end_word)
        print(">>> chunk word:")
        log_word("cunt")
        log_word("नग्न")
        print(">>> single word:")
        log_word("bitch")

    def test_check_single_word():
        # todo
        dfa = DFATree()
        dfa.init_from_file(os.path.dirname(__file__) + "/TestCase/watch_list.json", "json")
        # Chutiyapa
        t = dfa.root.children[" "].children["C"].children["h"].children["u"].children["t"].children["i"].children[
            "y"].children["a"].children["p"].children["a"].children[" "]


    # test_one()
    # test_init_from_json_file("बिस्तर गर्म, कामक्रीड़ा, सेक्सी, बिस्तर ग👌र्म")
    # test_init_from_json_file("ass")
    # test_case_from_file()
    # test_show_all_words()
    test_on_get_all_children_asList(True)
    test_check_wordChunk()




    # inpupt = sys.argv[1]
    # print("\n\n原文: ",inpupt)
    # print("\n判断的结果: ",dfa.search(inpupt))
    # dfa = DFATree()
    # dfa.init_from_file(os.path.dirname(__file__)+"/TestCase/watch_list.json","json")
    # for i in dfa.search(inpupt,return_json=False): print(i)

