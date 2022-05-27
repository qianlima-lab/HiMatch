# HiMatch  
The code for ACL-2021 Long Paper [Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classification](https://aclanthology.org/2021.acl-long.337)  


## Dependency  
```
PyTorch==1.4.0, sklearn, tqdm, transformers  
```

## Dataset  
[RCV1-V2](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)  
[WOS](https://github.com/kk7nc/HDLTex)  
[EURLEX-57K](https://github.com/iliaschalkidis/lmtc-eurlex57k)  
Glove.6B.300d.txt  

## Preprocess  
### Dataset Preprocess  
Transform your dataset to json format file {'token': List[str], 'label': List[str]}  
You can refer to data_modules/preprocess.py, and here is the WOS dataset [Google Drive](https://drive.google.com/file/d/1rOhTmMf6bgDOwLmAhIDdJwljgmuA0Tu0/view?usp=sharing) after preprocessing.  

### Label Prior Probability (Label Structure)  
Preprocess the taxnomy format (data/wos.taxnomy and data/wos_prob_child_parent.json)  
Extract Label Prior Probability  
```
python helper/hierarchy_tree_statistic.py config/wos.json  
```

### Label Description  
We use classic TD-IDF to extract the representative words for each label.  
```
python construct_label_desc.py  
```
In our follow-up actual practice, we found that introducing richer label representations is beneficial for further improvement.  

## Train  
Modify the training settings in config/wos.json.
```
python train.py config/wos-bert.json  
python train.py config/wos.json  
```
Hyperparamter Description  
```
sample_num: 2. The averge label number of WOS is 2. For every positive label, we all regard them as positive label index and construct matching pairs.  
negative_ratio: 3. Coarse-grained label, wrong sibling label and other wrong label.  
total_sample_num: 2*3=6.  
```

## Other Experimental Settings  
The experimental settings on EURLEX-57K: [KAMG](https://github.com/MemoriesJ/KAMG)  
The experimental settings on BERT: [Bert-Multi-Label-Text-Classification](https://github.com/lonePatient/Bert-Multi-Label-Text-Classification)  

## Cite  
```
@inproceedings{chen-etal-2021-hierarchy,
    title = "Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classification",
    author = "Chen, Haibin  and Ma, Qianli  and Lin, Zhenxi  and Yan, Jiangyue",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-long.337",
    pages = "4370--4379"
}
```