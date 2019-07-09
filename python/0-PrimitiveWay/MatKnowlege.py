# encoding=utf-8

#####
# 参考:https://www.jianshu.com/p/c7e642877b0e
#####
import numpy as np

# point size
m = 20

# points x-coordinate and dummy value
x0 = np.ones((m, 1))
x1 = np.arange(1, m + 1).reshape(m, 1)
X = np.concatenate((x0,x1), axis=1)
# points y-coordinate
y = np.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
              11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(m, 1)
# The Learning Rate
lr = 0.01
w = np.array([1, 1]).reshape(2, 1)

def error_function(theta, X, y):
    """
    Error function J definition.
    :param theta:
    :param X:
    :param y:
    :return:
    """
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    """
    Gradient of the function J definition.
    :param theta:
    :param X:
    :param y:
    :return:
    """
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha):
    """
    Perform gradient descent.
    :param X:
    :param y:
    :param alpha:
    :return:
    """
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

def compute_gradient(X,w,y):
    diff = np.dot(X, w) - y
    gradient = (1./m) * np.dot(X.T, diff)
    return gradient

def apply_gradient(w, gradient, lr):
    w_ = w - lr*gradient
    return w_


gradient = compute_gradient(X,w,y)
w = apply_gradient(w,gradient,lr)


diff = np.dot(X, w) - y
gradient = (1./m) * np.dot(X.T, diff) # 线性回归的导数解析式
i =0
while not np.all(np.absolute(gradient) <= 1e-5):
    theta = w - lr * gradient
    gradient = gradient_function(theta, X, y)
    if i % 1000 ==0 : print("gradient: %s" % ",".join(gradient))
    i += 1

optimal = w
# optimal = gradient_descent(X, y, lr)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])







