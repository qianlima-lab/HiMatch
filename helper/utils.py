#!/usr/bin/env python

import codecs
import torch
from models.structure_model.tree import Tree
import os
import json

def load_checkpoint(model_file, model, config, optimizer=None):
    """
    load models
    :param model_file: Str, file path
    :param model: Computational Graph
    :param config: helper.configure, Configure object
    :param optimizer: optimizer, torch.Adam
    :return: best_performance -> [Float, Float], config -> Configure
    """
    checkpoint_model = torch.load(model_file)
    config.train.start_epoch = checkpoint_model['epoch'] + 1
    best_performance = checkpoint_model['best_performance']
    model.load_state_dict(checkpoint_model['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_model['optimizer'])
    return best_performance, config


def save_checkpoint(state, model_file):
    """
    :param state: Dict, e.g. {'state_dict': state,
                              'optimizer': optimizer,
                              'best_performance': [Float, Float],
                              'epoch': int}
    :param model_file: Str, file path
    :return:
    """
    torch.save(state, model_file)


def get_hierarchy_relations(hierar_taxonomy, label_map, root=None, fortree=False):
    """
    get parent-children relationships from given hierar_taxonomy
    parent_label \t child_label_0 \t child_label_1 \n
    :param hierar_taxonomy: Str, file path of hierarchy taxonomy
    :param label_map: Dict, label to id
    :param root: Str, root tag
    :param fortree: Boolean, True : return label_tree -> List
    :return: label_tree -> List[Tree], hierar_relation -> Dict{parent_id: List[child_id]}
    """
    label_tree = dict()
    label_tree[0] = root
    hierar_relations = {}
    with codecs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                if fortree and parent_label == 'Root':
                    parent_label_id = -1
                else:
                    continue
            else:
                parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                                  for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
            if fortree:
                if "yelp" not in hierar_taxonomy and 'eurlex' not in hierar_taxonomy:
                    assert (parent_label_id + 1) in label_tree
                parent_tree = label_tree[parent_label_id + 1]

                for child in children_label_ids:
                    if "yelp" not in hierar_taxonomy and 'eurlex' not in hierar_taxonomy:
                        assert (child + 1) not in label_tree
                    child_tree = Tree(child)
                    parent_tree.add_child(child_tree)
                    label_tree[child + 1] = child_tree
    if fortree:
        return hierar_relations, label_tree
    else:
        return hierar_relations

def get_parent_sibling(hierar_taxonomy, label_map):
    """
    get parent-children relationships from given hierar_taxonomy
    parent_label \t child_label_0 \t child_label_1 \n
    :param hierar_taxonomy: Str, file path of hierarchy taxonomy
    :param label_map: Dict, label to id
    :return: hierar_relation -> Dict{child_id: parent_id}
    """
    hierar_relations = {}
    hierar_relations_sibling = {}
    with codecs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            #print(line)
            line_split = line.rstrip().split('\t')
            #print(line_split)
            parent_label, children_label = line_split[0], line_split[1:]
            #print(parent_label)
            #print(type(parent_label))
            #print(parent_label in label_map)
            if parent_label not in label_map:
                if parent_label != 'Root':
                    continue
            parent_label_id = parent_label
            children_label_ids = [child_label \
                                  for child_label in children_label if child_label in label_map]
            for child in children_label_ids:
                hierar_relations[child] = parent_label_id
                children_label_ids.remove(child)
                hierar_relations_sibling[child] = children_label_ids
                children_label_ids.append(child)
    return hierar_relations, hierar_relations_sibling

def get_parent(get_child, config, vocab):
    if config.data.dataset == "eurlex":
        hierarchy_prob_child_parent_file = os.path.join(config.data.data_dir, config.data.prob_child_to_parent_json)
        f = open(hierarchy_prob_child_parent_file, 'r')
        hierarchy_prob_str = f.readlines()
        f.close()
        hierarchy_prob_child_parent = json.loads(hierarchy_prob_str[0])
        hierarchy_prob_child_parent_id = {}
        for (child, parent) in hierarchy_prob_child_parent.items():
            if "Root" in parent or child not in vocab.v2i['label']:
                continue
            total_parent = []
            for p in parent:
                if p in vocab.v2i['label']:
                    total_parent.append(vocab.v2i['label'][p])
            if len(total_parent) != 0:
                hierarchy_prob_child_parent_id[vocab.v2i['label'][child]] = total_parent
    else:
        hierarchy_prob_child_parent_id = {}
        for (k, v) in get_child.items():
            for child in v:
                if child not in hierarchy_prob_child_parent_id:
                    hierarchy_prob_child_parent_id[child] = [k]
                else:
                    hierarchy_prob_child_parent_id[child].append(k)
    return hierarchy_prob_child_parent_id

def get_sibling(get_child, config, vocab):
    hierarchy_prob_sibling_id = {}
    for (k, v) in get_child.items():
        if len(v) == 1:
            continue
        for child in v:
            hierarchy_prob_sibling_id[child] = []
            for c in v:
                if c != child:
                    hierarchy_prob_sibling_id[child].append(c)

    if config.data.dataset == "eurlex":
        first_layer = []
        hierarchy_prob_child_parent_file = os.path.join(config.data.data_dir, config.data.prob_child_to_parent_json)
        f = open(hierarchy_prob_child_parent_file, 'r')
        hierarchy_prob_str = f.readlines()
        f.close()
        hierarchy_prob_child_parent = json.loads(hierarchy_prob_str[0])
        for (child, parent) in hierarchy_prob_child_parent.items():
            if "Root" in parent:
                first_layer.append(child)
        for child in first_layer:
            if len(first_layer) == 1:
                continue
            if child not in hierarchy_prob_sibling_id:
                hierarchy_prob_sibling_id[child] = []
                for c in first_layer:
                    if c != child:
                        hierarchy_prob_sibling_id[child].append(c)
    else:
        first_layer = []
        with codecs.open(os.path.join(config.data.data_dir, config.data.hierarchy), "r", "utf8") as f:
            for line in f:
                # print(line)
                line_split = line.rstrip().split('\t')
                parent_label, children_label = line_split[0], line_split[1:]
                if parent_label != "Root":
                    continue
                children_label = [vocab.v2i['label'][c] for c in children_label]
                first_layer = children_label
                if len(first_layer) == 1:
                    break
                for child in first_layer:
                    if child not in hierarchy_prob_sibling_id:
                        hierarchy_prob_sibling_id[child] = []
                        for c in first_layer:
                            if c != child:
                                hierarchy_prob_sibling_id[child].append(c)
                break
    return hierarchy_prob_sibling_id, first_layer