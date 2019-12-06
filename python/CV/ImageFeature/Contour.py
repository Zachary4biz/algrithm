# author: zac
# create-time: 2019-11-04 15:04
# usage: - 

"""
这个是轮廓检测
多个轮廓：https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html
一个轮廓：active_contour
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage
from skimage import color,data
from skimage.filters import gaussian
import skimage.segmentation as seg
from zac_pyutils import CVUtils
import math
from PIL import Image


# from skimage.draw import polygon
# poly = np.array((
#     (300, 300),
#     (480, 320),
#     (380, 430),
#     (220, 590),
#     (300, 300),
# ))
# rr, cc = polygon(poly[:,0], poly[:,1], img.shape)
# img[rr,cc,1] = 255

#fill polygon 参考资料:https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array

# create_polygon

def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """
    # resolution控制生成多少个点，可以理解为越多得到的圆就更圆
    radians = np.linspace(0, 2 * np.pi, resolution)
    c = center[1] + radius * np.cos(radians)  # polar co-ordinates
    r = center[0] + radius * np.sin(radians)
    return np.array([c, r]).T

def rectangle_points(center, w, h):
    points=[]
    w,h = int(w),int(h)
    # for col in [, center[0] + h//2]:
    #     for row in range(center[1] - w//2, center[1] + w//2):
    #         points.append([col, row])
    lt,rb = [center[1] - w//2, center[0] - h//2], [center[1] + w//2, center[0] + h//2]
    points += [[x, lt[1]] for x in range(lt[0], rb[0])]
    points += [[rb[0], y] for y in range(lt[1], rb[1])]
    points += [[x, rb[1]] for x in range(lt[0], rb[0])[::-1]]
    points += [[lt[0], y] for y in range(lt[1], rb[1])[::-1]]
    return np.array(points)

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(np.array(image), cmap=cmap)
    ax.axis('off')
    return fig, ax

def seg_image(img_inp, resolution=100, alpha=0.05, beta=0.15):
    width,height = img_inp.shape[1], img_inp.shape[0]  # skimage 读出的图像是按arr的行列来算的shape
    # R = min(width, height) * 0.45 # R=(w^2 + h^2)^0.5
    # points_ = circle_points(resolution, center=[height // 2,  width // 2], radius=R)[:-1]
    points_ = rectangle_points(center=[height // 2,  width // 2], w=width*0.9, h=height*0.9)
    snake_ = seg.active_contour(img_inp, points_, alpha=alpha, beta=beta)
    return snake_, points_

def plot_seg(img_inp, points, snake):
    fig, ax = image_show(img_inp)
    ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    lt, rb = (min(snake[:, 0]), min(snake[:, 1])), (max(snake[:, 0]), max(snake[:, 1]))
    rect = Rectangle((lt[0], lt[1]), rb[0] - lt[0], rb[1] - lt[1], linewidth=2, edgecolor='g', facecolor='none')
    # ax.add_patch(rect)
    fig.show()

if True:
    img = data.astronaut()
    # snake, points=seg_image(color.rgb2gray(img), resolution=140)
    snake, points=seg_image(img, resolution=140, alpha=0.3, beta=0.15)
    plot_seg(img, points=points, snake=snake)
    print(img.size)
    plt.show()
else:
    img_load = CVUtils.Load.image_by_pil_from("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3Wm23HuKYuKMiSo9U_UAFDYc1_ccodPS9PMNrOWesI3lAE0bF&s").convert("RGB")
    snake, points=seg_image(np.array(img_load), resolution=200, alpha=0.15, beta=0.05)
    # snake, points=seg_image(img, resolution=200)
    plot_seg(img_load, points=points, snake=snake)
    print(img_load.size)
    plt.show()





