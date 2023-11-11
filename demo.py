import numpy as np
import cv2
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 创建观测图像 Y 和观测掩码 O
img = cv2.imread('./image.png', cv2.IMREAD_GRAYSCALE)
img = img + np.random.normal(0, 20, img.shape)  # 加入高斯噪声
u = (img - np.min(img)) / (np.max(img) - np.min(img))  # 归一化处理

missing_ratio = 0.5
missing_mask = np.random.normal(0, 1, u.shape) > missing_ratio

Y = u.reshape(-1, 1)
O = missing_mask.reshape(-1, 1)

# 定义神经网络参数
input_size = Y.shape[1]
hidden_size = 128
output_size = Y.shape[1]
learning_rate = 0.01
lambda_value = 0.01

# 初始化权重和偏置
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_input_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_hidden_output = np.zeros((1, output_size))

# 存储损失函数和梯度范数
loss_history = []
grad_norm_history = []

# 训练神经网络
for epoch in range(1000):
    # 前向传播
    hidden_layer_input = np.dot(Y, weights_input_hidden) + bias_input_hidden
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden_output
    predicted_output = sigmoid(output_layer_input)

    # 计算损失函数
    regularization_term = np.linalg.norm(O * predicted_output, ord='fro') ** 2
    total_loss = 0.5 * np.linalg.norm(Y - O * predicted_output) ** 2 + lambda_value * regularization_term
    loss_history.append(total_loss)

    # 打印loss value
    print(f'第{epoch}步:loss={total_loss}')
    
    # 反向传播
    error = O * predicted_output * (1 - predicted_output)
    d_output = error * sigmoid_derivative(predicted_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * relu_derivative(hidden_layer_output)

    # 更新权重和偏置
    weights_hidden_output += learning_rate * hidden_layer_output.T.dot(d_output)
    bias_hidden_output += learning_rate * np.sum(d_output, axis=0, keepdims=True)
    weights_input_hidden += learning_rate * Y.T.dot(d_hidden)
    bias_input_hidden += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    # 计算梯度范数
    grad_norm = np.linalg.norm(np.concatenate([weights_hidden_output.flatten(), bias_hidden_output.flatten(),
                                               weights_input_hidden.flatten(), bias_input_hidden.flatten()]))
    grad_norm_history.append(grad_norm)

# 绘制损失函数和梯度范数的图像
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Total Loss')
plt.title('Total Loss Function')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(grad_norm_history, label='Gradient Norm')
plt.title('Gradient Norm')
plt.xlabel('Epoch')
plt.ylabel('Norm of Gradients')
plt.legend()

plt.show()
