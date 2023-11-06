import numpy as np

# 定义优化问题的目标函数和梯度函数
def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient_function(x):
    return np.array([2 * x[0], 2 * x[1]])

# 初始化参数和BB步长所需的变量
x = np.array([2.0, 3.0])  # 初始参数
max_iterations = 100  # 最大迭代次数
tolerance = 1e-6  # 收敛容忍度

x_prev = x.copy()
gradient_prev = gradient_function(x)  # 初始梯度

# 迭代优化过程
for iteration in range(max_iterations):
    gradient = gradient_function(x)  # 计算当前参数处的梯度
    if iteration > 0:
        s_k = x - x_prev
        y_k = gradient - gradient_prev
        step_length = np.dot(s_k, s_k) / np.dot(s_k, y_k)
    else:
        step_length = 1000  # 初始步长
    
    x_prev = x.copy()
    gradient_prev = gradient.copy()
    
    x = x - step_length * gradient  # 更新参数
    
    print(f"第{iteration}步:loss={objective_function(x)}")
    if np.linalg.norm(gradient) < tolerance:
        break

print("Optimal solution:", x)
print("Optimal value:", objective_function(x))
