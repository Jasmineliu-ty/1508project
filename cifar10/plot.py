import matplotlib.pyplot as plt

# 示例数据
# list1 = [0.974, 0.986, 0.9868, 0.9872, 0.9856, 0.9908]  # 第一组数据
# list2 = [0.9852, 0.9892, 0.99, 0.9884, 0.9888, 0.9852]  # 第二组数据
# list3 = [0.9852, 0.9876, 0.9856, 0.986, 0.988, 0.9884]  # 第三组数据
#
# list4 = [0.9704, 0.9868, 0.9892, 0.9864, 0.9876, 0.9872]  # 第三组数据
# list5 = [0.9728, 0.9856, 0.9844, 0.9864, 0.984, 0.9868]  # 第三组数据
#
# list6 = [0.9652, 0.9836, 0.9848, 0.986, 0.9872, 0.9848]  # 第三组数据
#
# list7 = [0.898, 0.9824, 0.9848, 0.9848, 0.9824, 0.9836]  # 第三组数据
# list8 = [0.9424, 0.9808, 0.9828, 0.9816, 0.9824, 0.9812]  # 第三组数据
#
# list9 = [0.9304, 0.9808, 0.9792, 0.9804, 0.982, 0.9812]  # 第三组数据
# list10 = [0.8936, 0.962, 0.978, 0.9744, 0.9812, 0.9736]  # 第三组数据
# # list3 = [0.9424, 0.9808, 0.9828, 0.9816, 0.9824, 0.9812]  # 第三组数据
# # list3 = [0.9424, 0.9808, 0.9828, 0.9816, 0.9824, 0.9812]  # 第三组数据

list1 = [0.7905, 0.817, 0.8441, 0.8422, 0.8602, 0.8656, 0.859, 0.8689, 0.8763, 0.8737]
list2 = [0.873, 0.875, 0.8815, 0.8889, 0.876, 0.888, 0.8814, 0.8861, 0.8904, 0.8854]
list3 = [0.8656, 0.8849, 0.883, 0.8881, 0.8837, 0.8816, 0.8845, 0.8947, 0.8823, 0.8924]
list4 = [0.8748, 0.8716, 0.8798, 0.8821, 0.8833, 0.8852, 0.8861, 0.8882, 0.8895, 0.8934]
list5 = [0.8568, 0.8705, 0.8569, 0.876, 0.8762, 0.878, 0.8779, 0.8793, 0.8736, 0.884]
#list6 =  [0.873, 0.8719, 0.88, 0.8862, 0.8825, 0.8813, 0.8882, 0.8947, 0.8863, 0.8861, 0.885, 0.8885, 0.8904, 0.8919, 0.8893]

# 这些列表表示不同实验或模型的准确率
all_data = [list1, list2, list3, list4, list5]
labels = ['87.87% filters', '75.75% filters','63.63% filters',
          '51.51% filters', '39.39% filters']  # 为每条曲线设置标签

# 生成 epoch 的横坐标
epochs = range(0, len(list1))

# 创建图像
plt.figure(figsize=(10, 6))

# 绘制每组数据
for i, data in enumerate(all_data):
    plt.plot(epochs, data, marker='o', label=labels[i])

# 添加标题和轴标签
plt.title('Training Accuracy vs Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# 显示图例
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图像
plt.tight_layout()
plt.show()
