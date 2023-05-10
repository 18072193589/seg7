#引用工具库
import pandas as pd
import matplotlib.pyplot as plt
#引用中文（如果不需要用到中文可以不写下面两行代码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#打开表
df = pd.read_csv(r"D:\Deep-Learing\NetModel\ganseg-master2.0\ganseg-master\hdb_to_oil_VG200_G_result_lr=0.001_3\hdb_to_oil_VG200_G\train_logs.csv")
#输入折线图数据
plt.subplot(221)
plt.plot(df[" epoch"],df[" G_recon_loss_A"],label='G_recon_loss_A',linewidth=1,color='c',marker='o',markerfacecolor='blue',markersize=1)
plt.xlabel("epoch")
plt.ylabel('loss')
plt.title("")
plt.legend()
plt.grid()


plt.subplot(222)
plt.plot(df[" epoch"],df[" G_recon_loss_B"],label='G_recon_loss_B',linewidth=1,color='y',marker='o',markerfacecolor='blue',markersize=1)
#横坐标为物品编号，纵坐标为库存量，线的名称为库存量，粗细为1，颜色为青色，标记为“o”所代表的图形（会在后面详细介绍），颜色为蓝色，大小为5
# plt.plot(df["iter"],df[" G_loss"],label='G_loss',linewidth=1,color='y',marker='o',markerfacecolor='blue',markersize=1)
# plt.plot(df["iter"],df[" U_loss_A2B"],label='U_loss_A2B',linewidth=1,color='r',marker='v',markerfacecolor='blue',markersize=1)
plt.xlabel(" epoch")
plt.ylabel('loss')
plt.title("")
plt.legend()
plt.grid()

plt.subplot(223)
plt.plot(df[" epoch"],df[" G_identity_loss_A"],label='G_identity_loss_A',linewidth=1,color='y',marker='o',markerfacecolor='blue',markersize=1)
#横坐标为物品编号，纵坐标为库存量，线的名称为库存量，粗细为1，颜色为青色，标记为“o”所代表的图形（会在后面详细介绍），颜色为蓝色，大小为5
# plt.plot(df["iter"],df[" G_loss"],label='G_loss',linewidth=1,color='y',marker='o',markerfacecolor='blue',markersize=1)
# plt.plot(df["iter"],df[" U_loss_A2B"],label='U_loss_A2B',linewidth=1,color='r',marker='v',markerfacecolor='blue',markersize=1)
plt.xlabel(" epoch")
plt.ylabel('loss')
plt.title("")
plt.legend()
plt.grid()

plt.subplot(224)
plt.plot(df[" epoch"],df[" G_identity_loss_B"],label='G_identity_loss_B',linewidth=1,color='y',marker='o',markerfacecolor='blue',markersize=1)
#横坐标为物品编号，纵坐标为库存量，线的名称为库存量，粗细为1，颜色为青色，标记为“o”所代表的图形（会在后面详细介绍），颜色为蓝色，大小为5
# plt.plot(df["iter"],df[" G_loss"],label='G_loss',linewidth=1,color='y',marker='o',markerfacecolor='blue',markersize=1)
# plt.plot(df["iter"],df[" U_loss_A2B"],label='U_loss_A2B',linewidth=1,color='r',marker='v',markerfacecolor='blue',markersize=1)
plt.xlabel(" epoch")
plt.ylabel('loss')
plt.title("")
plt.legend()
plt.grid()
# plt.subplot(223)
# plt.plot(df["epoch"],df[" U_loss_A2B"],label='U_loss_A2B',linewidth=1,color='r',marker='v',markerfacecolor='blue',markersize=1)
# plt.xlabel("iter")
# plt.ylabel('loss')
# plt.title("")
# plt.legend()
# plt.grid()
plt.show()
