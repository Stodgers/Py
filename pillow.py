from PIL import Image,ImageDraw,ImageColor,ImageFilter,ImageFont
import random

#读取文件
img =Image.open('1.jpg')
#获取图片大小
(w,h) = img.size
#裁切图片
box = (400,400,600,600)
region = img.crop(box)
region.save('crop.jpg')
#获取图片格式
imf = img.format
#图片模式的转换
img1 = img.convert("L")
#几何变换
rs = img.resize((800,600))
rz = img.rotate(45) #黑色填充旋转遗留区域
#rs.show()
#压缩图片
img.thumbnail((w*0.3,h*0.3))
#保存文件
img.save('1-1.jpg')


print(img.format,img.size,img.mode)




'''

#图片转字符
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
lens = len(ascii_char)
im = Image.open('wk.png')
#im.show()
(width,height) = im.size
im = im.resize((int(width*0.9),int(height*0.5)))
(width,height) = im.size
print(im.size)
def convertt(im):
    im = im.convert("L")
    txt = ""
    for i in range(height):
        for j in range(width):
            gray = im.getpixel((j,i))
            unit = 256.0 / lens
            txt += ascii_char[int(gray/unit)]
        txt += "\n"
    return txt

txt = convertt(im)
f = open("1.txt","w")
f.write(txt)
f.close()
'''
'''

im = Image.open("pdd.jpg")
draw = ImageDraw.Draw(im)
myFont = ImageFont.truetype('xg.ttf',size = 60)
co = 'pink'
(width,height) = im.size
draw.text((120,170),'萌萌哒',fill=co,font=myFont)
#im.show()
#draw.ellipse((width-40,0,width,40),fill='red',outline='red')
#draw.text((width-28,0),'1',fill='white',font=myFont)
im.save('pdd1.jpg')
'''

def rantxt():
    txt = []
    txt.append(random.randint(97,123))
    txt.append(random.randint(65,91))
    txt.append(random.randint(48,57))
    return chr(txt[random.randint(0,2)])

def tc():
    return (random.randint(64,255),random.randint(64,255),random.randint(64,255))
def bc():
    return (random.randint(32,127),random.randint(32,127),random.randint(32,127))
width = 240
height = 60
im = Image.new('RGB',(width,height),(255,255,255))

font = ImageFont.truetype(font='xg.ttf',size=40)
draw = ImageDraw.Draw(im)
for x in range(width):
    for y in range(height):
        draw.point((x,y),fill=bc())
#im.show()
for i in range(6):
    draw.text((40*i+15,5),rantxt(),fill=tc(),font=font)
#im = im.filter(ImageFilter.BLUR)
#im.show()
draw.ellipse((width-20,0,width,20),fill='red')
myfont = ImageFont.truetype('xg.ttf',15)
draw.text((width-14,0),'1',fill='white',font=myfont)
#im = im.filter(ImageFilter.BLUR)
im.show()

