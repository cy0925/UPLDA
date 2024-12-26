import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

csv_list = [os.path.join('./csv/', i) for i in os.listdir('./csv/') if i.endswith('.csv')]

# data = pd.read_csv('1.csv')

datas = [pd.read_csv(i) for i in csv_list]
names = [i.split('.')[1] for i in csv_list]
colomns = datas[0].columns.values

plt.figure(figsize=(12, 8))
for q,c in enumerate(colomns):
    plt.subplot(2,2,q+1)
    for qq in  range(len(datas)):
        plt.plot(datas[qq][c], label=names[qq])
    plt.title('%s'%c)
    plt.legend()
plt.savefig('1.png')

plt.figure(figsize=(12, 8))
for q,c in enumerate(colomns):
    plt.subplot(2,2,q+1)
    for qq in  range(len(datas)):
        plt.scatter(np.argmax(datas[qq][c]),np.max(datas[qq][c]), label=names[qq])
    plt.title('%s'%c)
    plt.legend()
plt.savefig('2.png')