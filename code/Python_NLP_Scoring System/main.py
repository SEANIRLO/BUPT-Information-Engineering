# -*- coding: utf-8 -*-
import requests
import time
import json
import re
from bs4 import BeautifulSoup
import os

import jieba.analyse
import numpy as np
import pandas as pd
#from transformers import BertModel, BertTokenizer, BertForMaskedLM
import tensorflow as tf

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from bert_serving.client import BertClient
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

'''
def filter_text(text):
    step1 = text.replace(' ', '')
    step2 = step1.replace('\n', '')
    step3 = step2.replace('\r','')
    step4 = step3.replace('\xa0','')
    step5 = step4.replace('\t', '')
    return step5

login = requests.session()
userinfo = {"user":"2019210277", "password":"qdzwcaex20001007"}
login.post("https://win.bupt.edu.cn/user.do?action=login", data=userinfo)

with open('ProjectID.txt','a',encoding='utf-8') as f:
    for classification in batch:
        page_start = 1
        page_end = 55
        for page in range(page_start, page_end + 1):
            r = requests.get('https://win.bupt.edu.cn/programlist.do?batch={index0}&cid=-1&key=&type=&directionid=&p={index1}.html'.format
                         (index0=classification,index1=page))
            if r.status_code == 200:
                print("Downloading page {index}".format(index=page))
                soup = BeautifulSoup(r.text, features="lxml")
            else:
                print("error")
                break
            # find basic info
            basic_info = str(soup.find(attrs={'id':'posts'}))
            iter = re.finditer(r'a href=\"program\.do\?id=(\d+)\"', basic_info)
            for id in iter:
                f.write(str(id[1])+'\n')
f.close()

id_all = []
student_pattern = re.compile(r'<i class=\"icon-([\w]*)\".*subtitle\">([\d]+)级 ([\u4e00-\u9fa5]+学院)',
                             re.MULTILINE | re.DOTALL)
tutor_pattern = re.compile(r'<i class=\"icon-([\w]*)\".*subtitle\">([\u4e00-\u9fa5]+学院|图书馆)',
                           re.MULTILINE | re.DOTALL)
score_pattern = re.compile(r'var eval_score = \[{\"score\":\"(\d+\.\d+)\",\"type\":\"1\"',
                           re.MULTILINE | re.DOTALL)
tag_pattern = re.compile(r'disabled=\"\">([\d\u4e00-\u9fa5\x21-\x7e]*)<\/button>',
                         re.MULTILINE | re.DOTALL)
counter = 0
written = 0
with open("projectID.txt", "r") as f:
    for id in f.readlines():
        id = id.strip('\n')
        id_all.append(id)
f.close()
for id in id_all:
    counter += 1
    r = requests.get('https://win.bupt.edu.cn/program.do?id={index}'.format
        (index=id))
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, features="lxml")
        basic_info = soup.find_all(attrs={'style': 'font-size:17px;line-height:25px;'})
        brief_intro = filter_text(re.findall(r'<div style=\"font-size:17px;line-height:25px;\">([\s\S]*)<\/div>',
                       str(basic_info[1]))[0])
    else:
        print("error")
        break
    if not brief_intro:
        continue
    elif not soup.find_all('script', text=score_pattern):
        continue
    else:
        info_dict = {}
        info_dict['id'] = id
        info_dict['score'] = re.findall(score_pattern, str(soup.find_all('script', text=score_pattern)))[0]
        buf0 = soup.find(attrs={'class': 'col-md-10 col-xs-12'})
        info_dict['topic'] = filter_text(re.search(r'<h2 style=\"display:inline\">([\s\S]*)<\/h2>', str(buf0))[1])
        info_dict['college'] = filter_text(re.search(r'<h3 style=\"display:inline;\">([\s\S]*)<\/h3>', str(buf0))[1])
        buf0_1 = re.findall(tag_pattern, str(buf0))
        info_dict['field'] = str(buf0_1[2])
        info_dict['theme'] = str(buf0_1[3])
        info_dict['intro'] = brief_intro
        buf1 = soup.find_all(attrs={'class': 'fancy-title title-dotted-border'})

        for people in buf1:
            if re.search(r'项目负责人', str(people.contents)):
                info_dict['director'] = re.findall(student_pattern, str(people.next_sibling.contents))
            elif re.search('项目成员', str(people.contents)):
                buf1_1 = []
                for sibling in people.next_siblings:
                    if re.findall(student_pattern, str(sibling)):
                        buf1_1 += re.findall(student_pattern, str(sibling.contents))
                info_dict['member'] = buf1_1
            elif re.search('指导老师', str(people.contents)):
                info_dict['tutor'] = re.findall(tutor_pattern, str(people.next_sibling.contents))
                info_dict['tutor_field'] = re.findall(tag_pattern,
                                             str(people.next_sibling.contents))
        # dict2json
        info_json = open("ProjectInfo.json", "a+", encoding='utf-8')
        json_text = json.dumps(info_dict, ensure_ascii=False)
        print(json_text)
        info_json.write(json_text+'\n')
        print("WriteIn successful")
        print('{:.2f}%'.format(counter/len(id_all)*100))
        time.sleep(0.1)
        info_json.close()
        written += 1
print(written)

counter = 0
data_dict = {}
with open('./ProjectInfo.json','r+',encoding='utf-8')as f:
    for line in f.readlines():
        counter += 1
        data = json.loads(line)
        #data = project.strip('\n')
        data_dict['id'] = data['id']
        data_dict['score'] = float(data['score'])
        data_dict['topic'] = data['topic']
        data_dict['college'] = data['college']
        data_dict['field'] = data['field']
        data_dict['theme'] = data['theme']
        data_dict['intro'] = data['intro']
        data_dict['director_gender'] = data['director'][0][0] 
        if 1734 <= int(data['id']) <= 2448:
            data_dict['director_grade'] = 2020 - int(data['director'][0][1])
        else:
            data_dict['director_grade'] = 2021 - int(data['director'][0][1])
        data_dict['director_college'] = data['director'][0][2]
        for member_id in range(4):
            if 'member' not in data:
                data_dict['member{index}_gender'.format(index=member_id + 1)] = 'nan'
                data_dict['member{index}_grade'.format(index=member_id + 1)] = 'nan'
                data_dict['member{index}_college'.format(index=member_id + 1)] = 'nan'
            else:
                if member_id<len(data['member']):
                    data_dict['member{index}_gender'.format(index=member_id + 1)] = data['member'][member_id][0]
                    if 1734<=int(data['id'])<=2448:
                        data_dict['member{index}_grade'.format(index=member_id+1)] = 2020 - int(data['member'][member_id][1])
                    else:
                        data_dict['member{index}_grade'.format(index=member_id + 1)] = 2021 - int(data['member'][member_id][1])
                    data_dict['member{index}_college'.format(index=member_id+1)] = data['member'][member_id][2]
                else:
                    data_dict['member{index}_gender'.format(index=member_id + 1)] = 'nan'
                    data_dict['member{index}_grade'.format(index=member_id+1)] = 'nan'
                    data_dict['member{index}_college'.format(index=member_id+1)] = 'nan'
        if data['tutor']:
            data_dict['tutor_gender'] = data['tutor'][0][0]
            data_dict['tutor_college'] = data['tutor'][0][1]
        # df.columns = ["id", "score", "topic", "college", "field", "theme", "intro",
        #              "director_gender", "director_grade", "director_college",
        #              "member1_gender", "member1_grade", "member1_college",
        #              "member2_gender", "member2_grade", "member2_college",
        #              "member3_gender", "member3_grade", "member3_college",
        #              "member4_gender", "member4_grade", "member4_college",
        #              "tutor_gender", "tutor_college",
        #              "tutor_field1", "tutor_field2", "tutor_field3"]
        for tag_id in range(3):
            if tag_id<len(data['tutor_field']):
                data_dict['tutor_field{index}'.format(index=tag_id+1)] = data['tutor_field'][tag_id]
            else:
                data_dict['tutor_field{index}'.format(index=tag_id+1)] = 'nan'
        df = pd.DataFrame(data_dict, index=[counter])
        print(df)
        print('{:.2f}%'.format(counter/1075*100))
        df.to_csv("Info.csv", encoding="utf-8", header=None, index=False, mode="a+")
        time.sleep(0.1)
f.close()
'''
'''
input = pd.read_csv('ProjectInfo.csv', header=None)
data = input.drop(columns=0)
data.fillna('无', inplace=True)
np.seterr(divide='raise', invalid='raise')

#ABOUT THE SCORE (LABEL)
score = np.array(data.iloc[:, 0])
for i in range(len(score)):
    score[i] = 100*(score[i]-60)/35+np.random.normal(0, 1, size=None)
    while score[i] < 0 or score[i] > 100:
        score[i] += np.random.normal(0, 5, size=None)
plt.hist(score)
plt.title("Histogram of Distribution: Uniformized Score with Gaussian Noise")
plt.show()

print('step 1 accomplished')

#ABOUT THE TEAM
#[电子工程(光电信息)，国际，计算机(软件)，经济管理，理，人文，数字媒体与设计艺术，网络空间安全，现代邮政(自动化)，信息与通信工程], 人工智能按比例分配至计软、信通、自邮, w_director = 1.5*w_member
people = np.zeros(shape=(1073,))
CDist = np.zeros(shape=(1073, 10)) #CDist for College Distribution
GAver = np.zeros(shape=(1073,)) #GAver for Grade Average
GRatio = np.zeros(shape=(1073,)) #GRatio for Gender Ratio (male serves as indicator)
collegeDist = np.array(pd.concat([data.iloc[:, 8], data.iloc[:, 11], data.iloc[:, 14], data.iloc[:, 17], data.iloc[:, 20]], axis=1))
for i in range(collegeDist.shape[0]):
    people[i] += 1
    if collegeDist[i, 0] == "电子工程学院": CDist[i, 0]+=1.5
    elif collegeDist[i, 0] == "光电信息学院": CDist[i, 0] += 1.5
    elif collegeDist[i, 0] == "国际学院": CDist[i, 1] += 1.5
    elif collegeDist[i, 0] == "计算机学院": CDist[i, 2] += 1.5
    elif collegeDist[i, 0] == "经济管理学院": CDist[i, 3] += 1.5
    elif collegeDist[i, 0] == "理学院": CDist[i, 4] += 1.5
    elif collegeDist[i, 0] == "人工智能学院":
        CDist[i, 2] += 0.525
        CDist[i, 8] += 0.45
        CDist[i, 9] += 0.525
    elif collegeDist[i, 0] == "人文学院": CDist[i, 5] += 1.5
    elif collegeDist[i, 0] == "软件学院": CDist[i, 2] += 1.5
    elif collegeDist[i, 0] == "数字媒体与设计艺术学院": CDist[i, 6] += 1.5
    elif collegeDist[i, 0] == "网络空间安全学院": CDist[i, 7] += 1.5
    elif collegeDist[i, 0] == "现代邮政学院": CDist[i, 8] += 1.5
    elif collegeDist[i, 0] == "信息与通信工程学院": CDist[i, 9] += 1.5
    elif collegeDist[i, 0] == "自动化学院": CDist[i, 8] += 1.5
    for j in range(1,5):
        if collegeDist[i, j] == '无' : continue
        else:
            if collegeDist[i, j] == "电子工程学院": CDist[i, 0] += 1
            elif collegeDist[i, j] == "光电信息学院": CDist[i, 0] += 1
            elif collegeDist[i, j] == "国际学院": CDist[i, 1] += 1
            elif collegeDist[i, j] == "计算机学院": CDist[i, 2] += 1
            elif collegeDist[i, j] == "经济管理学院": CDist[i, 3] += 1
            elif collegeDist[i, j] == "理学院": CDist[i, 4] += 1
            elif collegeDist[i, j] == "人工智能学院":
                CDist[i, 2] += 0.35
                CDist[i, 8] += 0.30
                CDist[i, 9] += 0.35
            elif collegeDist[i, j] == "人文学院": CDist[i, 5] += 1
            elif collegeDist[i, j] == "软件学院": CDist[i, 2] += 1
            elif collegeDist[i, j] == "数字媒体与设计艺术学院": CDist[i, 6] += 1
            elif collegeDist[i, j] == "网络空间安全学院": CDist[i, 7] += 1
            elif collegeDist[i, j] == "现代邮政学院": CDist[i, 8] += 1
            elif collegeDist[i, j] == "信息与通信工程学院": CDist[i, 9] += 1
            elif collegeDist[i, j] == "自动化学院": CDist[i, 8] += 1
            people[i] += 1
    CDist[i, :] /= people[i]+0.5
gradeDist = np.array(pd.concat([data.iloc[:, 7], data.iloc[:, 10], data.iloc[:, 13], data.iloc[:, 16], data.iloc[:, 19]], axis=1))
genderDist = np.array(pd.concat([data.iloc[:, 6], data.iloc[:, 9], data.iloc[:, 12], data.iloc[:, 15], data.iloc[:, 18]], axis=1))
for i in range(len(score)):
    male = 0
    for j in range(5):
        if gradeDist[i, j] == '无': continue
        else: GAver[i] += gradeDist[i, j]
        if genderDist[i, j] == '无': continue
        elif genderDist[i, j] == 'male': male += 1
    GAver[i] /= people[i]
    GRatio[i] = male/people[i]

print('step 2 accomplished')

#ABOUT THE PROJECT AND ITS CONSISTENCY WITH THE TUTOR
#[topic, field, theme, intro, tutor_field1, tutor_field2, tutor_field3]
bc = BertClient()
projInfo = np.array(pd.concat([data.iloc[:, 1], data.iloc[:, 3], data.iloc[:, 4], data.iloc[:, 5], data.iloc[:, 22], data.iloc[:, 23], data.iloc[:, 24]], axis=1))
projInfo[42][0] = '”后街“APP' #modify to run the program
projInfo[112][0] = '”邮你就行“资源平台' #modify to run the program
projInfo[229][0] = '至臻的茶' #modify to run the program
projInfo[413][0] = '”耙虫“爬虫软件' #modify to run the program
projInfo[594][0] = '”邮你订“订餐平台' #modify to run the program
projInfo[645][0] = '”易诸“租赁平台' #modify to run the program
projInfo[896][0] = '”动·物“动物科普AR软件' #modify to run the program
projCharac = np.zeros(shape=(1073, 3))#[introLength, topic_intro_similarity, project_tutor_consistency], similarity / consistency := cosine similarity
projData = np.zeros(shape=(1073, 768))
for i in range(len(score)):
    topicTag = jieba.analyse.extract_tags(projInfo[i][0], topK=3, withWeight=True)
    fieldTag = jieba.analyse.extract_tags(projInfo[i][1], topK=3, withWeight=True, allowPOS=('n', 'nt', 'nz', 'l'))
    introTag = jieba.analyse.extract_tags(projInfo[i][3], topK=3, withWeight=True, allowPOS=('n', 'nz', 'eng', 'vn'))

    buf1 = np.zeros(shape=(768,))
    buf2 = np.zeros(shape=(768,))
    buf3 = np.zeros(shape=(768,))
    buf4 = np.zeros(shape=(768,))
    weightSum1 = 0.0
    weightSum2 = 0.0
    weightSum3 = 0.0
    count = 0
    for j in range(len(topicTag)):
        buf1 += (bc.encode([topicTag[j][0]])).T.reshape(768,) * topicTag[j][1]
        weightSum1 += topicTag[j][1]
    for k in range(len(fieldTag)):
        buf2 += (bc.encode([fieldTag[k][0]])).T.reshape(768,) * fieldTag[k][1]
        weightSum2 += fieldTag[k][1]
    for l in range(len(introTag)):
        buf3 += (bc.encode([introTag[l][0]])).T.reshape(768,) * introTag[l][1]
        weightSum3 += introTag[l][1]
    for m in range(3):
        if projInfo[i][m+4] == '无': continue
        buf4 += bc.encode([projInfo[i][m+4]]).T.reshape(768,)
        count += 1
    topicVec = buf1 / weightSum1
    introVec = buf3 / weightSum3
    topic_intro_similarity = topicVec.dot(introVec) / (np.linalg.norm(topicVec) * np.linalg.norm(introVec))
    fieldVec = buf2 / weightSum2
    themeVec = (bc.encode([projInfo[i][2]])).T.reshape(768,)
    tutor_fieldVec = buf4 / count
    projVec = 0.15*topicVec + 0.15*fieldVec + 0.4*introVec + 0.3*themeVec
    project_tutor_consistency = projVec.dot(tutor_fieldVec) / (np.linalg.norm(projVec) * np.linalg.norm(tutor_fieldVec))
    projCharac[i][0] = len(projInfo[i][3])
    projCharac[i][1] = topic_intro_similarity
    projCharac[i][2] = project_tutor_consistency
    projData[i] = projVec

print('step 3 accomplished')

Dataset = np.concatenate((score[:, None], people[:, None], GAver[:, None], GRatio[:, None], CDist, projCharac, projData), axis=1)
print(Dataset, type(Dataset), Dataset.shape)
np.savetxt('Dataset.txt', Dataset, delimiter=' ', newline='\n', encoding='utf-8')
'''

data = np.loadtxt('Dataset.txt', encoding='utf-8')
y, x = np.array_split(data, [1], axis=1)
xMax = np.max(x, axis=0).reshape(784,1).T
xMin = np.min(x, axis=0).reshape(784,1).T
'''
x = (x-xMin)/(xMax-xMin)
y = y/100
yLabel = torch.from_numpy(y)
xVector = torch.from_numpy(x)
dataset = TensorDataset(xVector, yLabel)
#TrainSet:900, TestSet:173
test, train = torch.utils.data.random_split(dataset, [173, 900], generator=torch.Generator().manual_seed(0))
trainSet = DataLoader(train, batch_size=10, shuffle=True)
testSet = DataLoader(test, batch_size=5, shuffle=False)
'''
#Define NN
torch.set_default_tensor_type(torch.DoubleTensor)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(Net, self).__init__()
        self.input = nn.Sequential(nn.Linear(n_feature, n_hidden1), nn.LeakyReLU())   # input layer
        self.hidden1 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.LeakyReLU())  # hidden layer 1
        self.hidden2 = nn.Sequential(nn.Linear(n_hidden2, n_hidden3), nn.LeakyReLU())  # hidden layer 2
        self.predict = nn.Sequential(nn.Linear(n_hidden3, n_output),)   # output layer

    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.predict(x)
        return x
'''
model = Net(784, 200, 100, 10, 1)
print(model)

criterion = nn.MSELoss()  # this is for regression mean squared loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)


train_losses = []
eval_losses = []

#for m in model.modules():
#    if isinstance(m, nn.Linear):
#        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')

#Training Mode
for epoch in range(200):
    train_loss = 0
    model.train()

    for vector, label in trainSet:
        vector = Variable(vector)
        label = Variable(label)
        out = model(vector)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(trainSet))

#Validation Mode
    eval_loss = 0
    min_eval_loss = 1000
    model.eval()

    for vector, label in testSet:
        vector = Variable(vector)
        label = Variable(label)
        out = model(vector)
        loss = criterion(out, label)
        eval_loss += loss.item()
    eval_losses.append(eval_loss / len(testSet))
    print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'
          .format(epoch, train_loss / len(trainSet), eval_loss / len(testSet)))
    if eval_loss < min_eval_loss:
        min_eval_loss = eval_loss
        torch.save(model, 'C:\\Users\SEAN\PycharmProjects\DL+NLP\mymodel.pkl')

plt.title('Train & Validation Loss')
plt.plot(np.arange(len(train_losses)), train_losses)
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.show()
params = list(model.named_parameters())
for m in params:
    print(m)

model.eval()
predictions = []
for i in range(1073):
    inp = torch.from_numpy((x[i, :]-xMin)/(xMax-xMin))
    predictions.append(35*model(inp).item()+60)
print(predictions)
print(max(predictions), min(predictions))
'''
model = torch.load('C:\\Users\SEAN\PycharmProjects\DL+NLP\mymodel.pkl')

inputInfo = []
print("请输入您的项目名称\n")
inputInfo.append(input())
print("请输入您的项目领域（带有学科序号）\n")
inputInfo.append(input())
print("请输入您的项目类别\n")
inputInfo.append(input())
print("请输入您的项目简介\n")
inputInfo.append(input())
print("请输入该项目总学生数\n")
inputInfo.append(int(input()))
name = ["负责人", "成员1", "成员2", "成员3", "成员4"]
info = ["性别（男|女）", "年级（1|2|3|4）", "所在学院（全称，按照2020年院系调整前的院系结构+人工智能学院填写）"]
for i in range(5):
    for j in range(3):
        print(f"请输入该项目的{name[i]}的{info[j]}，若人员填写完毕则请输入\"无\"")
        buf = input()
        if buf=='无':
            if i==0: print("请填写成员！")
            inputInfo[4+3*i:4+3*i+3] = '无无无'
            break
        if j==1:
            inputInfo.append(int(buf))
        else: inputInfo.append(buf)
for k in range(3):
    print(f"请输入该项目的指导教师所关联的领域（{k+1}/3），若人员填写完毕则请输入\"无\"")
    inputInfo.append(input())
print(inputInfo)

collegeDist = np.array([inputInfo[7], inputInfo[10], inputInfo[13], inputInfo[16], inputInfo[19]])
people = inputInfo[4]
CDist = np.zeros(10,)
for i in range(people):
    if i==0:
        if collegeDist[i] == "电子工程学院": CDist[0]+=1.5
        elif collegeDist[i] == "光电信息学院": CDist[0] += 1.5
        elif collegeDist[i] == "国际学院": CDist[1] += 1.5
        elif collegeDist[i] == "计算机学院": CDist[2] += 1.5
        elif collegeDist[i] == "经济管理学院": CDist[3] += 1.5
        elif collegeDist[i] == "理学院": CDist[4] += 1.5
        elif collegeDist[i,] == "人工智能学院":
            CDist[2] += 0.525
            CDist[8] += 0.45
            CDist[9] += 0.525
        elif collegeDist[i] == "人文学院": CDist[5] += 1.5
        elif collegeDist[i] == "软件学院": CDist[2] += 1.5
        elif collegeDist[i] == "数字媒体与设计艺术学院": CDist[6] += 1.5
        elif collegeDist[i] == "网络空间安全学院": CDist[7] += 1.5
        elif collegeDist[i] == "现代邮政学院": CDist[8] += 1.5
        elif collegeDist[i] == "信息与通信工程学院": CDist[9] += 1.5
        elif collegeDist[i] == "自动化学院": CDist[8] += 1.5
    else:
        if collegeDist[i] == '无' : continue
        else:
            if collegeDist[i] == "电子工程学院": CDist[0] += 1
            elif collegeDist[i] == "光电信息学院": CDist[0] += 1
            elif collegeDist[i] == "国际学院": CDist[1] += 1
            elif collegeDist[i] == "计算机学院": CDist[2] += 1
            elif collegeDist[i] == "经济管理学院": CDist[3] += 1
            elif collegeDist[i] == "理学院": CDist[4] += 1
            elif collegeDist[i] == "人工智能学院":
                CDist[2] += 0.35
                CDist[8] += 0.30
                CDist[9] += 0.35
            elif collegeDist[i] == "人文学院": CDist[5] += 1
            elif collegeDist[i] == "软件学院": CDist[2] += 1
            elif collegeDist[i] == "数字媒体与设计艺术学院": CDist[6] += 1
            elif collegeDist[i] == "网络空间安全学院": CDist[7] += 1
            elif collegeDist[i] == "现代邮政学院": CDist[8] += 1
            elif collegeDist[i] == "信息与通信工程学院": CDist[9] += 1
            elif collegeDist[i] == "自动化学院": CDist[8] += 1
CDist /= people+0.5
gradeDist = np.array([inputInfo[6], inputInfo[9], inputInfo[12], inputInfo[15], inputInfo[18]])
genderDist = np.array([inputInfo[5], inputInfo[8], inputInfo[11], inputInfo[14], inputInfo[17]])
GAver= 0
GRatio = 0
male = 0
for i in range(5):
    if gradeDist[i] == '无': continue
    else:
        GAver += int(gradeDist[i])
        if genderDist[i] == '男': male += 1
GAver /= people
GRatio = male/people

bc = BertClient()
projInfo = np.array([inputInfo[0], inputInfo[1], inputInfo[2], inputInfo[3], inputInfo[19], inputInfo[20], inputInfo[21]])
projCharac = np.zeros(3,)#[introLength, topic_intro_similarity, project_tutor_consistency], similarity / consistency := cosine similarity

topicTag = jieba.analyse.extract_tags(projInfo[0], topK=3, withWeight=True)
fieldTag = jieba.analyse.extract_tags(projInfo[1], topK=3, withWeight=True, allowPOS=('n', 'nt', 'nz', 'l'))
introTag = jieba.analyse.extract_tags(projInfo[3], topK=3, withWeight=True, allowPOS=('n', 'nz', 'eng', 'vn'))

buf1 = np.zeros(shape=(768,))
buf2 = np.zeros(shape=(768,))
buf3 = np.zeros(shape=(768,))
buf4 = np.zeros(shape=(768,))
weightSum1 = 0.0
weightSum2 = 0.0
weightSum3 = 0.0
count = 0
for j in range(len(topicTag)):
    buf1 += (bc.encode([topicTag[j][0]])).T.reshape(768,) * topicTag[j][1]
    weightSum1 += topicTag[j][1]
for k in range(len(fieldTag)):
    buf2 += (bc.encode([fieldTag[k][0]])).T.reshape(768,) * fieldTag[k][1]
    weightSum2 += fieldTag[k][1]
for l in range(len(introTag)):
    buf3 += (bc.encode([introTag[l][0]])).T.reshape(768,) * introTag[l][1]
    weightSum3 += introTag[l][1]
for m in range(3):
    if projInfo[m+4] == '无': continue
    buf4 += bc.encode([projInfo[m+4]]).T.reshape(768,)
    count += 1
topicVec = buf1 / weightSum1
introVec = buf3 / weightSum3
topic_intro_similarity = topicVec.dot(introVec) / (np.linalg.norm(topicVec) * np.linalg.norm(introVec))
fieldVec = buf2 / weightSum2
themeVec = (bc.encode([projInfo[2]])).T.reshape(768,)
tutor_fieldVec = buf4 / count
projVec = 0.15*topicVec + 0.15*fieldVec + 0.4*introVec + 0.3*themeVec
project_tutor_consistency = projVec.dot(tutor_fieldVec) / (np.linalg.norm(projVec) * np.linalg.norm(tutor_fieldVec))
projCharac[0] = len(projInfo[3])
projCharac[1] = topic_intro_similarity
projCharac[2] = project_tutor_consistency
buffer = np.array([people, GAver, GRatio])
inputData = torch.tensor(np.concatenate([buffer, CDist, projCharac, projVec]))

processedData = (inputData-xMin)/(xMax-xMin)
model.eval()

result = model(processedData)
print("该项目的雏雁计划初审分数预测值是：{:.6f}，预测结果仅供参考。感谢您的使用！".format(60+35*result.item()))

