# -*- coinding:utf-8 -*-
from tkinter import *
from tkinter import scrolledtext
import urllib,requests,re
import threading #多线程
import tkinter.messagebox as messagebox
'''
def get(a):
    hd = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'}
    url = 'http://www.budejie.com/video/'+str(a)
    print(url)
    html = requests.get(url,headers = hd).text
    #print(html)
    rr0 = r'.html">(.*?)</a>'
    jg0 = re.compile(rr0)
    ans0 = re.findall(jg0, html)
    rr1 = r'data-mp4="(.*?)\.mp4'
    jg1 = re.compile(rr1)
    ans1 = re.findall(jg1,html)
    #for i in ans1 j in ans0:
        #print(j+':'+i+'.mp4')
    #print(ans1)
    LL = []
    tt = len(ans1)
    for i in range(tt):
        ss = (ans0[i]+':\n'+ans1[i]+'.mp4'+'\n')
        #print(ss)
        #LL.append(ss)
        print(ss)
        #text.insert(END,str(ss)+'\n\n')
def bushh():

    for i in range(1,10):
        get(i)
    varl.set('爬取完毕...')

root = Tk()
root.title('Clock')
root.geometry('+200+200')
#text = scrolledtext.ScrolledText(root,font = ('微软雅黑',10))
#text.grid()

varl = StringVar()
label = Label(root,font = ('微软雅黑',10),fg = 'red',textvariable = varl)
varl.set('已准备...')
label.grid()
button = Button(root,text = '开始爬取',font = ('微软雅黑',10),command = bushh)
button.grid()
root.mainloop()
'''
class Application(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.pack()
        self.createWidgets()
    '''
    def createWidgets(self):
        self.hellolabel = Label(self,text='Hello word!')
        self.hellolabel.pack()
        self.vvButton = Button(self,text='quit',command=self.quit)
        self.vvButton.pack()
    '''
    def createWidgets(self):
        self.nameInput = Entry(self)
        self.nameInput.pack()
        self.alertbutton = Button(self,text = 'Hello',command = self.hello)
        self.alertbutton.pack()
    def hello(self):
        name = self.nameInput.get() or 'word'
        messagebox.showinfo('Message', 'Hello, %s' % name)

'''
app = Application()
app.master.title('Hello worddddd!!!')
app.mainloop()
'''
def func(event):
    xin = Tk()
    xin.wm_title('Message')
    t = e.get()
    g = 'hello,%s'%t
    la = Label(xin,text=g)
    la.pack()
    xin.mainloop()

root = Tk()
root.wm_title('Hello 233')
e = Entry(root)
e.pack()
b1 = Button(root,text='233')
b1.bind('<Button-1>',func)
b1.pack()
root.mainloop()







