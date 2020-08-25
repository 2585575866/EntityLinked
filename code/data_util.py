import json
import pandas as pd
import os
from keras_bert import Tokenizer
import codecs
from trie import *
import random
import numpy as np


def entity_clear(entity):
    '''
    将一些特殊字符替换
    :param entity: 一个实体名字
    :return: 替换后的实体
    '''
    pun = {'，': ',',
           '·': '•',
           '：': ':',
           '！': '!',
           }
    for p in pun:
        if p in entity:
            entity=entity.replace(p,pun[p])
    return entity
def new_alias(kb_data_path,train_file_path):
    '''
    统计训练数据中不能链接到实体库的mentioin, 统计出现次数，将其添加到对应实体的别名中
    :return: 字典形式 key 为实体名字 value 为添加的新的别名字典
             如：'bilibili': {'b站', '哔哩哔哩', '哔哩哔哩弹幕视频网'}
    '''
    id_alias = {}
    entity_id={}
    id_entity={}
    with open(kb_data_path, 'r',encoding="utf-8") as f:
        for line in f:
            temDict = json.loads(line)
            subject = temDict['subject']
            subject_id = temDict['subject_id']
            alias = set()
            for a in temDict['alias']:
                alias.add(a)
                alias.add(a.lower())
            alias.add(subject.lower())
            alias.add(entity_clear(subject))
            id_alias[subject_id] = alias
            subject_id = temDict['subject_id']
            entity_name = set(alias)
            entity_name.add(subject)
            entity_name.add(subject.lower())
            for a in alias:
                entity_name.add(a.lower())
            id_entity[subject_id] = subject
            for n in entity_name:
                if n in entity_id:
                    entity_id[n].add(subject_id)
                else:
                    entity_id[n] = set()
                    entity_id[n].add(subject_id)
    with open(train_file_path,"r",encoding="utf-8") as f:
        entity_alias_num={}
        for line in f:
            temDict = json.loads(line)
            mention_data=temDict['mention_data']
            for men in mention_data:
                mention=men['mention']
                kb_id = men['kb_id']
                if kb_id != 'NIL':
                    if id_entity[kb_id]!=mention:
                        if mention not in id_alias[kb_id]:
                            if id_entity[kb_id] in entity_alias_num:
                                entity_alias_num[id_entity[kb_id]]['count'] +=1
                                if mention in entity_alias_num[id_entity[kb_id]]:
                                    entity_alias_num[id_entity[kb_id]][mention] += 1
                                else:
                                    entity_alias_num[id_entity[kb_id]][mention] = 1
                            else:
                                entity_alias_num[id_entity[kb_id]]={}
                                entity_alias_num[id_entity[kb_id]]['count']=1
                                entity_alias_num[id_entity[kb_id]][mention]=1
    entity_alias={}
    for en in entity_alias_num:
        total_num=entity_alias_num[en]['count']
        if total_num>4:
            entity_alias[en]=set()
            for alias in entity_alias_num[en]:
                if alias=='count':
                    continue
                a_num=entity_alias_num[en][alias]
                if a_num>3:
                    entity_alias[en].add(alias)
            if len(entity_alias[en])==0:
                entity_alias.pop(en)
    return entity_alias


def get_len(text_lens, max_len=510, min_len=30):
    """
    戒断过长文本你的长度，小于30不在戒断，大于30按比例戒断
    :param text_lens: 列表形式 data 字段中每个 predicate+object 的长度
    :param max_len: 最长长度
    :param min_len: 最段长度
    :return: 列表形式 戒断后每个 predicate+object 保留的长度
            如 input：[638, 10, 46, 9, 16, 22, 10, 9, 63, 6, 9, 11, 34, 10, 8, 6, 6]
             output：[267, 10, 36, 9, 16, 22, 10, 9, 42, 6, 9, 11, 31, 10, 8, 6, 6]

    """
    new_len = [min_len]*len(text_lens)
    sum_len = sum(text_lens)
    del_len = sum_len - max_len
    del_index = []
    for i, l in enumerate(text_lens):
        if l > min_len:
            del_index.append(i)
        else:
            new_len[i]=l
    del_sum = sum([text_lens[i]-min_len for i in del_index])
    for i in del_index:
        new_len[i] = text_lens[i] - int(((text_lens[i]-min_len)/del_sum)*del_len) - 1
    return new_len

def get_text(en_data,max_len=510,min_len=30):
    '''
    根据data字段数据生成描述文本，将 predicate项与object项相连，在将过长的依据规则戒断
    :param en_data: kb里面的每个实体的data数据
    :param max_len: 每个 predicate+object 的最大长度
    :param min_len: 每个 predicate+object 的最小长度
    :return: 每个实体的描述文本
    '''
    texts = []
    text = ''
    for data in en_data:
        texts.append(data['predicate'] + ':'+ data['object'] + '，')
    text_lens=[]
    for t in texts:
        text_lens.append(len(t))
    if sum(text_lens)<max_len:
        for t in texts:
            text=text+t
    else:
        new_text_lens=get_len(text_lens,max_len=max_len,min_len=min_len)
        for t,l in zip(texts,new_text_lens):
            text=text+t[:l]
    return text[:max_len]

def del_bookname(entity_name):
    '''
    删除书名号
    :param entity_name: 实体名字
    :return: 删除后的实体名字
    '''
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name

def kb_processing(kb_data_path,train_file_path,output_dir):
    '''
    知识库处理
    :return: 得到后续要用的一些文件具体为：
            entity_id字典  key:entity name  value: kb_id list
                        '胜利': ['10001', '19044', '37234', '38870', '40008', '85426', '86532', '140750']
            id_entity字典 key:kb_id value:subject(实体名字)
                         10001 胜利
            id_text字典 key：kb_id value:实体描述文本
                       '10001': '摘要:英雄联盟胜利系列皮肤是拳头公司制作的具有纪念意义限定系列皮肤之一。'
            id_type字典  key：kb_id value: entity type
                        '10001': ['Thing']

            type_index字典 key：type name value：index
                        ‘NAN’: 0
                        'Thing' :1
    '''
    new_entity_alias=new_alias(kb_data_path,train_file_path)
    id_text={}
    entity_id={}
    type_index={}
    type_index['NAN']=0
    type_i=1
    id_type={}
    id_entity={}

    with open(kb_data_path, 'r',encoding="utf-8") as f:

        for line in f:
            temDict = json.loads(line)
            subject=temDict['subject']
            subject_id=temDict['subject_id']
            alias = set()
            for a in temDict['alias']:
                alias.add(a)
                alias.add(a.lower())
            alias.add(subject.lower())
            alias.add(entity_clear(subject))
            if subject in new_entity_alias:
                alias=alias|new_entity_alias[subject]
            en_data=temDict['data']
            en_type=temDict['type']
            entity_name=set(alias)
            entity_name.add(subject)
            for t in en_type:
                if not t in type_index:
                    type_index[t]=type_i
                    type_i+=1
            for n in entity_name:
                n=del_bookname(n)
                if n in entity_id:
                    entity_id[n].append(subject_id)
                else:
                    entity_id[n]=[]
                    entity_id[n].append(subject_id)
            id_type[subject_id]=en_type
            text=get_text(en_data)
            id_text[subject_id]=text
            id_entity[subject_id]=subject

    pd.to_pickle(entity_id,os.path.join(output_dir,"entity_id.pkl"))
    pd.to_pickle(id_entity, os.path.join(output_dir,"id_entity.pkl"))
    pd.to_pickle(type_index,  os.path.join(output_dir,"type_index.pkl"))
    pd.to_pickle(id_type,  os.path.join(output_dir,"id_type.pkl"))
    pd.to_pickle(id_text,  os.path.join(output_dir,"id_text.pkl"))

def get_token_dict():
    bert_path = '../bert_model/'
    dict_path = bert_path+'vocab.txt'
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

def get_link_entity(mention,id,entity_id,id_entity):
    '''
    1. 先通过mention 得到要消歧的所有实体id，
    2. 弱通过mention 找不到，则通过对应id，根据id找到对应实体，然后在找到要消歧的所有实体id，
    :param mention: 训练集中mention
    :param id: mention对应id
    :param entity_id: 实体名字对应所有实体的id
    :param id_entity: id对应的实体名字
    :return:去掉正确实体id的，id列表
    '''
    link_entitys=[]
    if mention in entity_id:
        link_entitys+=list(entity_id[mention])
        link_entitys+=list(entity_id[id_entity[id]])
    else:
        link_entitys+=list(entity_id[id_entity[id]])
    link_entitys=set(link_entitys)
    link_entitys.remove(id)
    link_entitys=list(link_entitys)
    random.shuffle(link_entitys)
    return link_entitys[:3]

def get_link_entity_test(mention,entity_id):
    if mention in entity_id:
        return list(entity_id[mention])
    return []



def get_train_input():
    '''
    消歧输入的构建
    :return:
    '''
    id_type=pd.read_pickle('../data/id_type.pkl')
    type_index=pd.read_pickle('../data/type_index.pkl')
    entity_id = pd.read_pickle('../data/entity_id.pkl')
    id_entity=pd.read_pickle('../data/id_entity.pkl')
    id_text=pd.read_pickle('../data/id_text.pkl')
    inputs = {'ids': [], 'seg': [],'begin':[],'end':[],'en_type':[],'labels':[]}
    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    file_len=0
    input_file = '../data/entity_id.pkl'
    trie_obj=get_Trie(input_file)
    with open('../rawData/train.json',encoding="utf-8") as f:
        for line in f:
            if file_len%100==0:
                print(file_len)
            file_len+=1
            temDict = json.loads(line)
            text=temDict['text']
            match_en = trie_obj.search_entity(text)
            mention_data=temDict['mention_data']
            for men in mention_data:
                mention=men['mention']
                kb_id=men['kb_id']
                offset = men['offset']
                begin=int(offset)+1
                end = begin+len(mention)
                if kb_id != 'NIL':
                    link_id=[kb_id]
                    link_id+=get_link_entity(mention,kb_id,entity_id,id_entity)

                    for id in link_id:
                        kb_text = id_text[id]
                        kb_type=type_index[id_type[id][0]]
                        indice, segment = tokenizer.encode(first=text,second=kb_text,max_len=256)
                        inputs['ids'].append(indice)
                        inputs['seg'].append(segment)
                        inputs['begin'].append([begin])
                        inputs['end'].append([end])
                        if id ==kb_id:
                            inputs['labels'].append(1)
                        else:
                            inputs['labels'].append(0)
                        inputs['en_type'].append([kb_type])
            mention_set=set()
            for men in mention_data:
                mention_set.add((men['mention'],int(men['offset'])))
            for en in match_en:
                if not en in mention_set:

                    link_id = get_link_entity_test(en[0],entity_id)
                    for id in link_id[:1]:
                        kb_text = id_text[id]
                        kb_type = type_index[id_type[id][0]]
                        indice, segment = tokenizer.encode(first=text, second=kb_text, max_len=256)
                        inputs['ids'].append(indice)
                        inputs['seg'].append(segment)
                        inputs['begin'].append([begin])
                        inputs['end'].append([end])
                        inputs['labels'].append(0)
                        inputs['en_type'].append([kb_type])
                    break

    for k in inputs:
        inputs[k]=np.array(inputs[k])
        print(k,inputs[k].shape)
        print(inputs[k][1])
    pd.to_pickle(inputs,'../data/train_input_bert_final.pkl')


def get_infer_input(input_file, out_file):
    id_type = pd.read_pickle('../data/id_type.pkl')
    type_index = pd.read_pickle('../data/type_index.pkl')
    entity_id = pd.read_pickle('../data/entity_id.pkl')

    id_text = pd.read_pickle('../data/id_text.pkl')

    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    out_file = open(out_file, 'w')
    file_index = 0
    with open(input_file) as f:
        for line in f:
            if file_index%100==0:
                print(file_index)
            file_index+=1

            temDict = json.loads(line)
            text = temDict['text']
            mention_data = temDict['mention_data']
            for men in mention_data:
                mention = men['mention']

                offset = int(men['offset'])
                begin = int(offset)+1
                end = begin + len(mention)

                link_id = get_link_entity_test(mention, entity_id)
                men['link_id'] = link_id
                link_data = {'ids': [], 'seg': [],'begin':[],'end':[],'en_type':[]}
                for id in link_id:

                    kb_text = id_text[id]
                    kb_type = type_index[id_type[id][0]]
                    indice, segment = tokenizer.encode(first=text, second=kb_text, max_len=256)
                    link_data['ids'].append(indice)
                    link_data['seg'].append(segment)
                    link_data['begin'].append([begin])
                    link_data['end'].append([end])
                    link_data['en_type'].append([kb_type])
                men['link_data'] = link_data

            out_file.write(json.dumps(temDict, ensure_ascii=False))
            out_file.write('\n')



if __name__ == '__main__':


    ## 1.对原始数据库数据和训练数据进行处理
    # kb_data_path="../rawData/kb_data"
    # train_file_path = '../rawData/train.json'
    # output_dir = "../data"
    # kb_processing(kb_data_path,train_file_path,output_dir)

    ## 2.获得实体链接的训练语料
    get_train_input()
    ## 3.获得线上推断时的输入数据
    #../data/eval_ner_result.json 为通过其它模型得到的命名实体结果，
    #../data/eval_link_binary_bert.json 为转换后的实体链接的输入
    get_infer_input('../data/eval_ner_result.json', '../data/eval_link_binary_bert.json')

