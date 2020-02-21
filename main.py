
# coding: utf-8

# In[ ]:


#用户在entry组件输入文字，决定在哪里插入，然后所有的文字在下方的text对话框中打印
from tkinter import *
import tkinter
#初始化模型
import bilsm_crf_model
import process_data
import numpy as np
import tensorflow as tf
import os

#设置GPU
import keras.backend.tensorflow_backend as KTF
#配置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # 使用编号为1，2号的GPU

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# KTF.set_session(sess)

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
model.load_weights('./checkpoint/model_020.weights')
predict_text = '阳虚血瘀证【概述】阳虚血瘀证是指阳气不足，血行瘀滞所表现的证候。'
def get_result(predict_text):
    str, length = process_data.process_data(predict_text, vocab)
    raw = model.predict(str)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]
    b_ill,b_med,b_pre,b_pul,b_sym,b_syn,b_tim = "","","","","","",""
    for s, t in zip(predict_text, result_tags):
        if t in ('B-ILL', 'I-ILL'):
            b_ill += ' ' + s if (t == 'B-ILL') else s
        if t in ('B-MED', 'I-MED'):
            b_med += ' ' + s if (t == 'B-MED') else s
        if t in ('B-PRE', 'I-PRE'):
            b_pre += ' ' + s if (t == 'B-PRE') else s
        if t in ('B-PUL', 'I-PUL'):
            b_pul += ' ' + s if (t == 'B-PUL') else s
        if t in ('B-SYM', 'I-SYM'):
            b_sym += ' ' + s if (t == 'B-SYM') else s
        if t in ('B-SYN', 'I-SYN'):
            b_syn += ' ' + s if (t == 'B-SYN') else s
        if t in ('B-TIM', 'I-TIM'):
            b_tim += ' ' + s if (t == 'B-TIM') else s
    #med是中药，pre是方剂，pul是脉象，sym是症状，syn是证候，tim是舌像，ill我觉得应
#     return "b_ill"+b_ill+"\nb_med"+b_med+"\nb_pre"+b_pre+"\nb_pul"+b_pul+"\nb_sym"+b_sym+"\nb_syn"+b_syn+"\nb_tim"+b_tim
    result = ""
    #疾病类术语，证候类术语，症状类术语，舌像类术语，脉象类术语，方剂类术语，中药类术语
    name_list = ["疾病类术语","中药类术语","方剂类术语","脉象类术语","症状类术语","证候类术语","舌像类术语"]
    
    value_list = [b_ill,b_med,b_pre,b_pul,b_sym,b_syn,b_tim]
    
    return value_list
    

    
# 插入函数（insert），在索隐处插入文字,
def insert_end():
    # 获取entry输入文字    
    string = entry1.get()
    if(len(string)<=0):
        text1.delete('1.0','end')  
        text1.insert("end","不允许为空，请重新输入！！！")
        return
    result = get_result(string)
    # 在text对象结尾插入文字
    text1.delete('1.0','end')    
    display_text = "医药实体识别：\n"+"原句子："+string+"\n"
    text1.insert("end",display_text)
    for r,e in zip(result,entry_list):
        e.delete('0','end')
        e.insert("end", r)

    
        
    
    
root = Tk()
root.title("基于Python的中医药术语抽取系统")
root.minsize(100, 100)

label1 = tkinter.Label(root, text="请输入：",width = 10)
entry1 = Entry(root,width = 30)


button2 = Button(root, text="确定", command=insert_end)
text1 = Text(root,width=45,height=10)

label1.grid(row=0, column=0,sticky=tkinter.NSEW,pady=5)
entry1.grid(row=0, column=1,sticky=tkinter.NSEW,pady=5)

button2.grid(row=0, column=2,sticky=tkinter.NSEW,pady=5)

text1.grid(row=2,columnspan=3,padx=20,pady=5)

ill_name = ["疾病类术语","中药类术语","方剂类术语","脉象类术语","症状类术语","证候类术语","舌像类术语"]
entry_list = list()
index = 0
split_line = "=============医药术语抽取结果============="
split_label = tkinter.Label(root, text=split_line)
split_label.grid(row=3,columnspan=3)

for ill_n in ill_name:
    label = tkinter.Label(root, text=ill_n,width = 10)
    label.grid(row=4+index,column = 0)
    entry = Entry(root,width = 30)
    entry.grid(row=4+index,column = 1,columnspan=2)
    entry_list.append(entry)
    index+=1
    
root.mainloop()

