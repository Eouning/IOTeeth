from function import *
import os

if __name__ == '__main__':
    print("<++=初始参数=++>")
    repetition=20
    print("repetition:",repetition)
    dir_name='position1/'
    print("dir_name:",dir_name)
    model_position="models/5_183_96.78456115722656.pth"
    print("model_position:",model_position)

    print("\n\n<++=使用示例1=++>\n------单张图片识别")
    img=cv2.imread("1.jpg")
    result=run(img,model_position,repetition)
    print(result)

    
    print("\n\n<++=使用示例2=++>\n------图片列表识别")
    img_list= os.listdir(dir_name)
    img_list.sort()
    img_list=[cv2.imread(dir_name+x) for x in img_list]
    result=list_run(img_list,model_position,repetition)
    print(result)

    print("\n\n<++=使用示例3=++>\n------在循环中的识别")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    let=Small().to(device)
    model_load(let,model_position)
    for img in img_list:
        rl=[]
        for x in range(repetition):
            rl.append(testimg(img,let))
        result=statistics.mode(rl)
        print(result)

    print("\n\n<++=使用示例4=++>\n------输出权重的识别")
    img=cv2.imread("1.jpg")
    result=run0(img,model_position)
    print(result)