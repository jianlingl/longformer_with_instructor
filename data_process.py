import json
import os, sys, gc, psutil
from datasets import Dataset
import six, numbers

bad_data_ids_arxiv = [line.strip() for line in open("data_clear/arxiv_train_bad_items.txt", 'r')]
bad_data_ids_pubmed = [line.strip() for line in open("data_clear/pubmed_train_bad_items.txt", 'r')]

def load_from_dict(path):
    if "train" in path:
        data_dict = read_into_memory_as_dict(path)

    if "val" in path:
        data = Dataset.from_dict(data_dict).select(range(325))
    else:
        data = Dataset.from_dict(data_dict)

    # release memory
    print("before del: ", sys.getrefcount(data_dict))
    print("memory used before del : %4f GB" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))
    del data_dict
    gc.collect()
    print("memory used after del : %4f GB" % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))


    print("data from dict processing finished")
    return data

def read_into_memory_as_dict_4all(infile):
    dict_in_memory = {}
    num_articles = sum([1 for _ in open(infile)])
    print("/".join(infile.split('/')[-2:]), " total data number: ", str(num_articles))
    idx = 0
    bad_data = []
    abstract_str_list = []
    section_names_list = []
    article_list = []
    if 'arxiv' in infile:
        bad_data = bad_data_ids_arxiv
    elif 'pubmed' in infile:
        bad_data = bad_data_ids_pubmed
    else:
        print("no pubmed & arxiv here")
    bad_data_num = 0

    for line in open(infile):
        idx += 1
        if not line.strip():
            continue
        line = line.strip()
        data = json.loads(line)
        if data['article_id'] in bad_data:
            bad_data_num += 1
            pass
        abstract_str = _list_to_string(data['abstract_text'])
        article = data['article_text']
        # add section names
        section_names = [e if e else 'None' for e in data['section_names']]
        section_names = _list_to_string(section_names)

        # add all the information into corresponding list
        abstract_str_list.append(abstract_str.replace("</S>", "").replace("<S>", ""))
        article_list.append(' '.join(article))
        section_names_list.append(section_names)

        # if idx % 5 == 0:
        # 	print('Finished reading {:.3f}\% of {:d} articles.'.format(
        # 		idx * 100.0 / num_articles, num_articles), end='\r', flush=True)
    if len(article_list) != len(abstract_str_list) or len(section_names_list) != len(article_list):
        print("wrong account")
    print('bad data in ', infile, ' account: ', str(bad_data_num))
    dict_in_memory['abstract'] = abstract_str_list
    dict_in_memory['article'] = article_list
    dict_in_memory['section_names'] = section_names_list
    return dict_in_memory

def read_into_memory_as_dict_4only_extract(infile):
    dict_in_memory = {}
    num_articles = sum([1 for _ in open(infile)])
    print("/".join(infile.split('/')[-2:]), " total data in extraxted set: ", str(num_articles))
    idx = 0
    abstract_str_list = []
    section_names_list = []
    extract_list = []

    for line in open(infile):
        idx += 1
        if not line.strip():
            continue
        line = line.strip()
        data = json.loads(line)
        abstract_str = _list_to_string(data['abstract'])
        extracts = data['extract']
        # add section names
        section_names = data['section_names'].replace('<EI/>', ',')

        # add all the information into corresponding list
        abstract_str_list.append(abstract_str.replace("</S>", "").replace("<S>", ""))
        extract_list.append(extracts.replace("</S>", "").replace("<S>", ""))
        section_names_list.append(section_names)

    # if idx % 5 == 0:
    # 	print('Finished reading {:.3f}\% of {:d} articles.'.format(
    # 		idx * 100.0 / num_articles, num_articles), end='\r', flush=True)
    if len(extract_list) != len(abstract_str_list) or len(section_names_list) != len(extract_list):
        print("wrong account")

    dict_in_memory['abstract'] = abstract_str_list
    dict_in_memory['article'] = extract_list
    dict_in_memory['section_names'] = section_names_list
    return dict_in_memory

def read_into_memory_as_dict_4all_extract(infile):
    dict_in_memory = {}
    num_articles = sum([1 for _ in open(infile)])
    print("/".join(infile.split('/')[-2:]), " total data in extraxted set: ", str(num_articles))
    idx = 0
    abstract_str_list = []
    section_names_list = []
    extract_list = []

    for line in open(infile):
        idx += 1
        if not line.strip():
            continue
        line = line.strip()
        data = json.loads(line)
        abstract_str = _list_to_string(data['abstract'])
        extracts = data['extract']
        # add section names
        section_names = data['section_names'].replace('<EI/>', ',')

        # add all the information into corresponding list
        abstract_str_list.append(abstract_str.replace("</S>", "").replace("<S>", ""))
        extract_list.append(extracts.replace("</S>", "").replace("<S>", ""))
        section_names_list.append(section_names)

    # if idx % 5 == 0:
    # 	print('Finished reading {:.3f}\% of {:d} articles.'.format(
    # 		idx * 100.0 / num_articles, num_articles), end='\r', flush=True)
    if len(extract_list) != len(abstract_str_list) or len(section_names_list) != len(extract_list):
        print("wrong account")

    dict_in_memory['abstract'] = abstract_str_list
    dict_in_memory['article'] = extract_list
    dict_in_memory['section_names'] = section_names_list
    return dict_in_memory

def _list_to_string(lst):
    ret = ''
    if not lst:
        return ret
    if isinstance(lst[0], six.string_types):
        ret = ','.join(lst)
    elif isinstance(lst[0], numbers.Number):
        ret = ','.join([str(e) for e in lst])
    else:
        print(type(lst[0]))
        raise AttributeError('Unacceptable format of list to return to string')
    return ret

