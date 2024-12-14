import matplotlib.pyplot as plt

# 示例数据
list1 = [0.974, 0.986, 0.9868, 0.9872, 0.9856, 0.9908]  # 第一组数据
list2 = [0.9852, 0.9892, 0.99, 0.9884, 0.9888, 0.9852]  # 第二组数据
list3 = [0.9852, 0.9876, 0.9856, 0.986, 0.988, 0.9884]  # 第三组数据

list4 = [0.9704, 0.9868, 0.9892, 0.9864, 0.9876, 0.9872]  # 第三组数据
list5 = [0.9728, 0.9856, 0.9844, 0.9864, 0.984, 0.9868]  # 第三组数据

list6 = [0.9652, 0.9836, 0.9848, 0.986, 0.9872, 0.9848]  # 第三组数据

list7 = [0.898, 0.9824, 0.9848, 0.9848, 0.9824, 0.9836]  # 第三组数据
list8 = [0.9424, 0.9808, 0.9828, 0.9816, 0.9824, 0.9812]  # 第三组数据

list9 = [0.9304, 0.9808, 0.9792, 0.9804, 0.982, 0.9812]  # 第三组数据
list10 = [0.8936, 0.962, 0.978, 0.9744, 0.9812, 0.9736]  # 第三组数据
# list3 = [0.9424, 0.9808, 0.9828, 0.9816, 0.9824, 0.9812]  # 第三组数据
# list3 = [0.9424, 0.9808, 0.9828, 0.9816, 0.9824, 0.9812]  # 第三组数据

# 这些列表表示不同实验或模型的准确率
all_data = [list1, list2, list3, list4, list5, list6, list7, list8, list9, list10]
labels = ['93.94% filters', '87.88% filters','81.82% filters',\
          '75.76% filters','69.70% filters','63.64% filters',\
            '57.58% filters','51.52% filters','45.45% filters','39.39% filters',]  # 为每条曲线设置标签

# 生成 epoch 的横坐标
epochs = range(0, len(list1))

# 创建图像
plt.figure(figsize=(10, 6))

# 绘制每组数据
for i, data in enumerate(all_data):
    plt.plot(epochs, data, marker='o', label=labels[i])

# 添加标题和轴标签
# plt.title('Training Accuracy vs Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# 显示图例
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图像
plt.tight_layout()
plt.show()
