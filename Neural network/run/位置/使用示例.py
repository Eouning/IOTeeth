from function import *
import os

if __name__ == '__main__':
    print("<++=使用示例1=++>\n------单张图片识别")
    img=cv2.imread("1.jpg")
    result=run(img,"models/4分类|成功率95.4792.pth")
    print(result)

    print("\n\n<++=使用示例2=++>\n------图片列表识别")
    img_list= os.listdir("img_list")
    img_list.sort()
    img_list=[cv2.imread("img_list/"+x) for x in img_list]
    result=list_run(img_list,"models/4分类|成功率95.4792.pth")
    print(result)

    print("\n\n<++=使用示例3=++>\n------在循环中的识别")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    let=Small().to(device)
    model_load(let,"models/4分类|成功率95.4792.pth")
    for img in img_list:
        result=testimg(img,let)
        print(result)