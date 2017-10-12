# encoding=utf-8
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# 完全随机的生成数据
def generate_random():
    print ("完全随机地生成数据")
    x, y = make_classification(n_samples=100,
                               n_features=20,
                               n_informative=2,
                               n_redundant=2,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=2,
                               weights=None,
                               flip_y=0.01,
                               class_sep=1.0,
                               hypercube=True,
                               shift=0.0,
                               scale=1.0,
                               shuffle=True,
                               random_state=None)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color="blue", marker="x")
    plt.scatter(x[y == 0, 0], x[y == 0, 1], color="#4458A7", marker="o")
    plt.show()


# 生成圆形数据
def generate_circle():
    x, y = make_circles(n_samples=500,
                        shuffle=True,
                        noise=0.2,  # 0~1之间可以控制分布圆不圆
                        random_state=None,
                        factor=0.8)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color="blue", marker="x")
    plt.scatter(x[y == 0, 0], x[y == 0, 1], color="#4458A7", marker="o")
    plt.show()


# 生成月牙形状的数据
def generate_moon():
    # 非常整齐的两个月牙
    x1, y1 = make_moons(n_samples=100,
                        shuffle=True,
                        noise=None,
                        random_state=None)
    # 加入噪音,但还看得出来月牙形状
    x, y = make_moons(n_samples=500,
                      shuffle=True,
                      noise=0.3,
                      random_state=1)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color="blue", marker="x")
    plt.scatter(x[y == 0, 0], x[y == 0, 1], color="#4458A7", marker="o")
    plt.show()


def main():
    # 通常noise可以控制生成的分布够不够圆、够不够月牙
    # 完全随机地生成数据
    # generate_random()
    # generate_circle()
    generate_moon()


if __name__ == '__main__':
    main()
