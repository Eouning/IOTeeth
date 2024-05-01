from function import *

if __name__ == '__main__':
    img=cv2.imread("1.jpg")
    result=run(img,"models/5_337_95.18071746826172.pth")
    print(result)