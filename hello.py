from PIL import Image,ImageDraw,ImageColor,ImageFilter     #导入模块
import random
import sys,os
im = Image.open('1.jpg')   #打开图片
print(im.format,im.size,im.mode)   #打印图片格式尺寸
#im.thumbnail((200, 100))
#im.save('3.jpg','JPEG')#保存图片，  到这个文件夹下就有压缩后的图片了
#print('Done!')
#

'''
#读取文件
im = Image.open('wk.png')

#获取图片大小
(width,height) = im.size
#获取图片格式
imf = im.format
#图片模式的转换
im = im.convert('L')
#压缩图片
im.thumbnail((width*0.3,height*0.3))
#保存文件
im.save("1-1.jpg")
#获取每个坐标的像素点的RGB值

im.show()
'''
'''
(w,h) = im.size
for i in range(w):
    for j in range(h):
        gray = im.getpixel((i, j))
        print(gray)
#重设图片大小

'''





ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
length = len(ascii_char)
img = Image.open('1-1.jpg')      #读取图像文件
(width,height) = img.size
img = img.resize((int(width*0.9),int(height*0.5)))  #对图像进行一定缩小
print(img.size)
def convert(img):
    img = img.convert("L")  # 转为灰度图像
    txt = ""
    for i in range(img.size[1]):
        for j in range(img.size[0]):
            gray = img.getpixel((j, i))     # 获取每个坐标像素点的灰度
            unit = 256.0 / length
            txt += ascii_char[int(gray / unit)] #获取对应坐标的字符值
        txt += '\n'
    return  txt

def convert1(img):
    txt = ""
    for i in range(img.size[1]):
        for j in range(img.size[0]):
            r,g,b = img.getpixel((j, i))           #获取每个坐标像素点的rgb值
            gray = int(r * 0.299 + g * 0.587 + b * 0.114)   #通过灰度转换公式获取灰度
            unit = (256.0+1)/length
            txt += ascii_char[int(gray / unit)]  # 获取对应坐标的字符值
        txt += '\n'
    return txt

txt = convert1(img)
f = open("03_convert.txt","w")
f.write(txt)            #存储到文件中
f.close()
