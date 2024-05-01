from function import *

#需修改的参数
learn_rate=1e-4         #学习率，防止过拟合
epochs=10000            #训练轮数
batch_size=5            #数据集的划分量
shuffle=True            #生成数据集时是否打乱
save_dir="models/"      #保存模型的文件夹
data_dir="datas/"       #存放训练数据集的文件夹

if __name__ == '__main__':
    if data_dir[-1]!='/':
        data_dir=data_dir+'/'

    if save_dir[-1]!='/':
        save_dirr=save_dir+'/'

    print("训练数据加载成功")

    train_loader,test_loader=get_loader('datas/images','datas/images_val','datas/label','datas/label_val',batch_size,shuffle)

    print("数据集加载成功")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    let=ResNet34().to(device)

    print("网络新建成功")

    #可选加载初始的模型权重
    """"
    model_load(let,"models/72.0989761352539.pth")

    print("原始权重加载成功")
    """

    print("训练开始！！！")

    train(let,train_loader,test_loader,learn_rate,epochs,batch_size,save_dir)