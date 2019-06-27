# encoding=utf-8
####
#
####
class BTNode(object):
    def __init__(self, value=None, left=None, right=None):
        self.left = left
        self.right = right
        self.value = value


class BTree(object):
    def __init__(self):
        # BTNode型
        self.root = None

    # 用循环实现
    # 基本思路就是添加的时候,先考虑左边的
    # 每次先检查左节点再检查右节点,如果都已有值则继续往下一层查
    # 如查完根节点后,会检查 "根的左节点" 是否还有左右节点,然后检查 "根的右节点" 是否还有左右节点
    # 然后检查 "根的左节点的左节点",然后检查 "根的左节点的右节点" ,然后检查 "根的右节点的左节点",然后检查 "根的右节点的右节点"
    def add_loop_way(self, v):
        node = BTNode(value=v)
        if self.root is None:
            # 若树还没有根节点,则把添加的值放入根节点
            self.root=node
        else:
            # 树已经有根节点,把根节点放入数组中
            q = [self.root]
            while True:
                # 从根节点开始依次遍历获取到数组q的第一个元素
                pop_node = q.pop(0)
                if pop_node.left is None:
                    # 该节点(第一次为根节点)的左节点为空,则直接赋给左节点
                    pop_node.left = node
                    # 并终止循环
                    return
                elif pop_node.right is None:
                    # 该节点(第一次为根节点)的右节点为空,则直接赋给右节点
                    pop_node.right = node
                    # 并终止循环
                    return
                else:
                    # 该节点(第一次为根节点)的左右节点都加入数组q中
                    q.append(pop_node.left)
                    q.append(pop_node.right)

    # 用递归实现
    def add_recursion_way(self,v):
        pass

    # 层次遍历
    def traverse(self):
        if self.root is None:
            return None
        # 根节点
        q = [self.root]
        # 根节点的值
        res = [self.root.value]
        while q != []:
            # 从根节点开始遍历,循环内的append操作每次加入的顺序是先加入左节点(如果有左节点的话)然后加入右节点
            pop_node = q.pop(0)
            if pop_node.left is not None:
                # 如果有左节点,就把左节点和左节点的值 加入两个数组中
                q.append(pop_node.left)
                res.append(pop_node.left.value)
            if pop_node.right is not None:
                # 如果有右节点...
                q.append(pop_node.right)
                res.append(pop_node.right.value)
        return res

    # todo: 先序、中序、后序s

# ----------todo: 答案 ----------------
#  def preorder(self,root):  # 先序遍历
#         if root is None:
#             return []
#         result = [root.item]
#         left_item = self.preorder(root.child1)
#         right_item = self.preorder(root.child2)
#         return result + left_item + right_item
#
#     def inorder(self,root):  # 中序序遍历
#         if root is None:
#             return []
#         result = [root.item]
#         left_item = self.inorder(root.child1)
#         right_item = self.inorder(root.child2)
#         return left_item + result + right_item
#
#     def postorder(self,root):  # 后序遍历
#         if root is None:
#             return []
#         result = [root.item]
#         left_item = self.postorder(root.child1)
#         right_item = self.postorder(root.child2)
#         return left_item + right_item + result
#
#     t = Tree()
# for i in range(10):
#     t.add(i)
# print('层序遍历:',t.traverse())
# print('先序遍历:',t.preorder(t.root))
# print('中序遍历:',t.inorder(t.root))
# print('后序遍历:',t.postorder(t.root))
