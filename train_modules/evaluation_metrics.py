#!/usr/bin/env python
# coding:utf-8

import numpy as np
from sklearn.metrics import accuracy_score


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    acc_score_list = []
    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        acc_up = len(set(sample_gold).intersection(set(sample_predict_id_list)))
        acc_down = len(set(sample_gold).union(set(sample_predict_id_list)))
        acc_score_list.append(acc_up / acc_down)
        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                    precision_micro + recall_micro) > 0 else 0.0

    #print("acc: ", sum(acc_score_list) / len(acc_score_list))
    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}

def evaluate_matching(epoch_predicts, epoch_labels, threshold=0.5):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    epoch_predicts = np.array(epoch_predicts)
    epoch_labels = np.array(epoch_labels)
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_predicts = np.array(epoch_predicts).reshape((-1))
    epoch_labels = np.array(epoch_labels).reshape((-1))
    epoch_predicts[epoch_predicts > threshold] = 1
    epoch_predicts[epoch_predicts <= threshold] = 0

    acc = 0
    for i in range(len(epoch_labels)):
        if epoch_labels[i] == epoch_predicts[i]:
            acc += 1
    acc = acc / len(epoch_labels)
    return {'acc': acc}

def evaluate_rcv_layer(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0
    right_total_layer3, predict_total_layer3, gold_total_layer3 = 0, 0, 0

    rcv1_layer1 = ["CCAT", "ECAT", "GCAT", "MCAT"]
    rcv1_layer2 = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C21', 'C22', 'C23', 'C24', 'C31', 'C32',
                   'C33', 'C34', 'C41', 'C42', 'E11', 'E12', 'E13', 'E14', 'E21', 'E31', 'E41', 'E51', 'E61', 'E71',
                   'G15', 'GCRIM', 'GDEF', 'GDIP', 'GDIS', 'GENT', 'GENV', 'GFAS', 'GHEA', 'GJOB', 'GMIL', 'GOBIT',
                   'GODD', 'GPOL', 'GPRO', 'GREL', 'GSCI', 'GSPO', 'GTOUR', 'GVIO', 'GVOTE', 'GWEA', 'GWELF', 'M11',
                   'M12', 'M13', 'M14']

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_layer1:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        elif label in rcv1_layer2:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]
        else:
            right_total_layer3 += right_count_list[i]
            gold_total_layer3 += gold_count_list[i]
            predict_total_layer3 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    precision_micro_layer3 = float(right_total_layer3) / predict_total_layer3 if predict_total_layer3 > 0 else 0.0
    recall_micro_layer3 = float(right_total_layer3) / gold_total_layer3
    micro_f1_layer3 = 2 * precision_micro_layer3 * recall_micro_layer3 / (
            precision_micro_layer3 + recall_micro_layer3) if (
                                                                     precision_micro_layer3 + recall_micro_layer3) > 0 else 0.0

    # l1-Macro-F1
    fscore_dict_l1 = [v for k, v in fscore_dict.items() if k in rcv1_layer1]
    macro_f1_layer1 = sum(fscore_dict_l1) / len(fscore_dict_l1)

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k in rcv1_layer2]
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    # l3-Macro-F1
    fscore_dict_l3 = [v for k, v in fscore_dict.items() if k not in rcv1_layer1 and k not in rcv1_layer2]
    macro_f1_layer3 = sum(fscore_dict_l3) / len(fscore_dict_l3)
    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'fscore_dict': fscore_dict,
            'l1_micro_f1': micro_f1_layer1,
            'l2_micro_f1': micro_f1_layer2,
            'l3_micro_f1': micro_f1_layer3,
            'l1_macro_f1': macro_f1_layer1,
            'l2_macro_f1': macro_f1_layer2,
            'l3_macro_f1': macro_f1_layer3}


def evaluate_wos_layer(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0

    rcv1_layer1 = ["CS", "Medical", "Civil", "ECE", "biochemistry", "MAE", "Psychology"]
    rcv1_layer2 = "other"

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_layer1:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        else:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    # l1-Macro-F1
    fscore_dict_l1 = [v for k, v in fscore_dict.items() if k in rcv1_layer1]
    macro_f1_layer1 = sum(fscore_dict_l1) / len(fscore_dict_l1)

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k not in rcv1_layer1]
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'l1_micro_f1': micro_f1_layer1,
            'l2_micro_f1': micro_f1_layer2,
            'l1_macro_f1': macro_f1_layer1,
            'l2_macro_f1': macro_f1_layer2
            }


def evaluate_math_layer(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list]

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0
    right_total_layer3, predict_total_layer3, gold_total_layer3 = 0, 0, 0
    right_total_layer4, predict_total_layer4, gold_total_layer4 = 0, 0, 0

    rcv1_layer1 = ['1176326814415724544', '1176327025812840448', '1176326884557070336', '1176326977637064704',
                   '1176326911555805184']
    rcv1_layer2 = ['1176327112077090816', '1176327510267535360', '1176327467137507328', '1176327551161999360',
                   '1176327374044930048', '1176327148504621056', '1176327432593219584', '1176327353807413248',
                   '1176327329526587392', '1176327587170099200', '1176327246953324544', '1176329010112897024',
                   '1176328976935952384', '1176329055835004928', '1176329104652509184', '1176331222201409536',
                   '1176328941892542464', '1176328921801826304', '1176328518028763136', '1176328401812987904',
                   '1176328440421556224']
    rcv1_layer3 = ['1176329262698078208', '1176329211514986496', '1176330387283255296', '1176330412478439424',
                   '1176330271746957312', '1176330330689511424', '1176330233121611776', '1176330198933839872',
                   '1176330646482853888', '1176330616606826496', '1176329985124999168', '1176329337478324224',
                   '1176329396802560000', '1176330155883503616', '1176330109389643776', '1176329955790036992',
                   '1176329848046755840', '1176329885334118400', '1176329745793818624', '1176329788043042816',
                   '1176330702942380032', '1176330753466966016', '1176329543687086080', '1176329710620385280',
                   '1176329651484893184', '1176329475902939136', '1176333165682499584', '1176333135760334848',
                   '1176333094786179072', '1176332755995467776', '1176332824903688192', '1176332934324690944',
                   '1176333276336627712', '1176333246020198400', '1176333304631402496', '1176333419177844736',
                   '1176333366262505472', '1176331414615105536', '1176328191527362560', '1176328368791232512',
                   '1176328314412081152', '1176332504010072064', '1176332538164289536', '1176332305564966912',
                   '1176332348011323393', '1176332203135868928', '1176332230432399360', '1176332269766582272',
                   '1176331882586185728', '1176331976379211776', '1176331687546855424', '1176331650112692224',
                   '1176331812352565248', '1176331768337539072']

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_layer1:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        elif label in rcv1_layer2:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]
        elif label in rcv1_layer3:
            right_total_layer3 += right_count_list[i]
            gold_total_layer3 += gold_count_list[i]
            predict_total_layer3 += predicted_count_list[i]
        else:
            right_total_layer4 += right_count_list[i]
            gold_total_layer4 += gold_count_list[i]
            predict_total_layer4 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    precision_micro_layer3 = float(right_total_layer3) / predict_total_layer3 if predict_total_layer3 > 0 else 0.0
    recall_micro_layer3 = float(right_total_layer3) / gold_total_layer3
    micro_f1_layer3 = 2 * precision_micro_layer3 * recall_micro_layer3 / (
            precision_micro_layer3 + recall_micro_layer3) if (
                                                                     precision_micro_layer3 + recall_micro_layer3) > 0 else 0.0

    precision_micro_layer4 = float(right_total_layer4) / predict_total_layer4 if predict_total_layer4 > 0 else 0.0
    recall_micro_layer4 = float(right_total_layer4) / gold_total_layer4
    micro_f1_layer4 = 2 * precision_micro_layer4 * recall_micro_layer4 / (
            precision_micro_layer4 + recall_micro_layer4) if (
                                                                     precision_micro_layer4 + recall_micro_layer4) > 0 else 0.0

    fscore_layer1 = [v for k, v in fscore_dict.items() if k in rcv1_layer1]
    macro_f1_layer1 = sum(fscore_layer1) / len(fscore_layer1)

    fscore_layer2 = [v for k, v in fscore_dict.items() if k in rcv1_layer2]
    macro_f1_layer2 = sum(fscore_layer2) / len(fscore_layer2)

    fscore_layer3 = [v for k, v in fscore_dict.items() if k not in rcv1_layer3]
    macro_f1_layer3 = sum(fscore_layer3) / len(fscore_layer3)

    fscore_layer4 = [v for k, v in fscore_dict.items() if
                     k not in rcv1_layer1 and k not in rcv1_layer2 and k not in rcv1_layer3]
    macro_f1_layer4 = sum(fscore_layer4) / len(fscore_layer4)
    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'l1_micro_f1': micro_f1_layer1,
            'l2_micro_f1': micro_f1_layer2,
            'l3_micro_f1': micro_f1_layer3,
            'l4_micro_f1': micro_f1_layer4,
            'l1_precision': precision_micro_layer1,
            'l2_precision': precision_micro_layer2,
            'l3_precision': precision_micro_layer3,
            'l4_precision': precision_micro_layer4,
            'l1_macro_f1': macro_f1_layer1,
            'l2_macro_f1': macro_f1_layer2,
            'l3_macro_f1': macro_f1_layer3,
            'l4_macro_f1': macro_f1_layer4
            }


def evaluate_rcv_tail(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0

    rcv1_head = ['CCAT', 'GCAT', 'MCAT', 'C15', 'ECAT', 'M14', 'C151', 'C152', 'GPOL', 'M13', 'M141', 'C18', 'M11',
                 'E21', 'C181', 'C17', 'GCRIM', 'GVIO', 'C31', 'GDIP']
    rcv1_tail = ['G156', 'G159', 'E313', 'GFAS', 'E142', 'E141', 'GOBIT', 'E61', 'E132', 'GTOUR', 'C331', 'E143',
                 'G152', 'GSCI', 'G155', 'E411', 'C313', 'E311', 'C32', 'G157']

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_head:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        elif label in rcv1_tail:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    # l1-Macro-F1
    fscore_dict_l1 = [v for k, v in fscore_dict.items() if k in rcv1_head]
    macro_f1_layer1 = sum(fscore_dict_l1) / len(fscore_dict_l1)

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k in rcv1_tail]
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'fscore_dict': fscore_dict,
            'head_micro_f1': micro_f1_layer1,
            'tail_micro_f1': micro_f1_layer2,
            'head_macro_f1': macro_f1_layer1,
            'tail_macro_f1': macro_f1_layer2}


def evaluate_math_tail(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0

    rcv1_head = ['1176326814415724544', '1176327025812840448', '1176327246953324544', '1176331222201409536',
                 '1176326884557070336', '1176326911555805184', '1176328401812987904', '1176329475902939136',
                 '1176327329526587392', '1176329055835004928', '1176329788043042816', '1176326977637064704',
                 '1176333276336627712', '1176327510267535360', '1176331414615105536', '1176331687546855424',
                 '1186906659990282377', '1176331650112692224', '1176327112077090816', '1176329262698078208']
    rcv1_tail = ['1186906659990282260', '1186906659990282362', '1186906659990282245', '1186906659990282392',
                 '1186906659990282313', '1186906659990282357', '1186906659990282256', '1186906659990282412',
                 '1186906659990282387', '1186906659990282275', '1186906659990282356', '1186906659990282282',
                 '1186906659990282251', '1186906659990282308', '1186906659990282324', '1186906659990282393',
                 '1186906659990282371', '1186906659990282363', '1186906659990282262', '1186906659990282302']

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_head:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        elif label in rcv1_tail:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    fscore_layer1 = [v for k, v in fscore_dict.items() if k in rcv1_head]
    macro_f1_layer1 = sum(fscore_layer1) / len(fscore_layer1)

    fscore_layer2 = [v for k, v in fscore_dict.items() if k in rcv1_tail]
    macro_f1_layer2 = sum(fscore_layer2) / len(fscore_layer2)

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'head_micro_f1': micro_f1_layer1,
            'tail_micro_f1': micro_f1_layer2,
            'head_macro_f1': macro_f1_layer1,
            'tail_macro_f1': macro_f1_layer2
            }

def evaluate_yelp_tail(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0

    tail_index = np.load('./data/yelp_tailindex.npy')
    rcv1_tail = []
    for i in tail_index:
        rcv1_tail.append(vocab.i2v['label'][i])

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_tail:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k in rcv1_tail]
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'fscore_dict': fscore_dict,
            'tail_micro_f1': micro_f1_layer2,
            'tail_macro_f1': macro_f1_layer2}

def evaluate_wos_tail(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))]
    right_count_list = [0 for _ in range(len(label2id.keys()))]
    gold_count_list = [0 for _ in range(len(label2id.keys()))]
    predicted_count_list = [0 for _ in range(len(label2id.keys()))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict)
        sample_predict_id_list = []
        if top_k is None:
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                sample_predict_id_list.append(sample_predict_descent_idx[j])

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    right_total_layer1, predict_total_layer1, gold_total_layer1 = 0, 0, 0
    right_total_layer2, predict_total_layer2, gold_total_layer2 = 0, 0, 0

    rcv1_head = ['Medical', 'Psychology', 'CS', 'biochemistry', 'ECE', 'Civil', 'MAE', 'Molecular biology',
                 'Polymerase chain reaction', 'Northern blotting']
    rcv1_tail = ['Structured Storage', 'Voltage law', 'Lorentz force law', 'Kidney Health', 'Digestive Health',
                 'Senior Health', 'Psoriasis', 'Healthy Sleep', 'Stealth Technology', 'Sprains and Strains']

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]
        if label in rcv1_head:
            right_total_layer1 += right_count_list[i]
            gold_total_layer1 += gold_count_list[i]
            predict_total_layer1 += predicted_count_list[i]
        elif label in rcv1_tail:
            right_total_layer2 += right_count_list[i]
            gold_total_layer2 += gold_count_list[i]
            predict_total_layer2 += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (
                                                                                                precision_micro + recall_micro) > 0 else 0.0

    precision_micro_layer1 = float(right_total_layer1) / predict_total_layer1 if predict_total_layer1 > 0 else 0.0
    recall_micro_layer1 = float(right_total_layer1) / gold_total_layer1
    micro_f1_layer1 = 2 * precision_micro_layer1 * recall_micro_layer1 / (
            precision_micro_layer1 + recall_micro_layer1) if (
                                                                     precision_micro_layer1 + recall_micro_layer1) > 0 else 0.0

    precision_micro_layer2 = float(right_total_layer2) / predict_total_layer2 if predict_total_layer2 > 0 else 0.0
    recall_micro_layer2 = float(right_total_layer2) / gold_total_layer2
    micro_f1_layer2 = 2 * precision_micro_layer2 * recall_micro_layer2 / (
            precision_micro_layer2 + recall_micro_layer2) if (
                                                                     precision_micro_layer2 + recall_micro_layer2) > 0 else 0.0

    # l1-Macro-F1
    fscore_dict_l1 = [v for k, v in fscore_dict.items() if k in rcv1_head]
    macro_f1_layer1 = sum(fscore_dict_l1) / len(fscore_dict_l1)

    # l2-Macro-F1
    fscore_dict_l2 = [v for k, v in fscore_dict.items() if k not in rcv1_tail]
    macro_f1_layer2 = sum(fscore_dict_l2) / len(fscore_dict_l2)

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'head_micro_f1': micro_f1_layer1,
            'tail_micro_f1': micro_f1_layer2,
            'head_macro_f1': macro_f1_layer1,
            'tail_macro_f1': macro_f1_layer2
            }


def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos


def mean_recall_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def mean_ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    Mean NDCG @k : float
    """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)


def evaluate_recall_ndcg(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None, label_map=None, dataset="eurlex"):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_gold = []
    epoch_predict = []
    epoch_gold_frequent = []
    epoch_predict_frequent = []
    epoch_gold_few = []
    epoch_predict_few = []
    epoch_gold_zero = []
    epoch_predict_zero = []

    frequent_mask = np.array([False] * len(label_map))
    few_mask = np.array([False] * len(label_map))
    zero_mask = np.array([False] * len(label_map))

    with open('./data/'+dataset+'_label_type.txt') as f:
        type_dic = eval(f.readline())
    for l in type_dic['frequent']:
        frequent_mask[label_map[l]] = True
    for l in type_dic['few']:
        few_mask[label_map[l]] = True
    for l in type_dic['zero']:
        zero_mask[label_map[l]] = True

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_labels):
        np_sample_gold = np.zeros((len(sample_predict)), dtype=np.float32)
        for l in sample_gold:
            np_sample_gold[l] = 1
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        epoch_predict.append(np_sample_predict)
        epoch_gold.append(np_sample_gold)

        epoch_gold_frequent.append(np_sample_gold[frequent_mask])
        epoch_gold_few.append(np_sample_gold[few_mask])
        epoch_gold_zero.append(np_sample_gold[zero_mask])

        epoch_predict_frequent.append(np_sample_predict[frequent_mask])
        epoch_predict_few.append(np_sample_predict[few_mask])
        epoch_predict_zero.append(np_sample_predict[zero_mask])

    # epoch_gold = epoch_gold[:3]
    # epoch_predict = epoch_predict[:3]
    recall5 = mean_recall_k(epoch_gold, epoch_predict, top_k)
    ndcg5 = mean_ndcg_score(epoch_gold, epoch_predict, top_k)

    recall5_frequent = mean_recall_k(epoch_gold_frequent, epoch_predict_frequent, top_k)
    ndcg5_frequent = mean_ndcg_score(epoch_gold_frequent, epoch_predict_frequent, top_k)

    recall5_few = mean_recall_k(epoch_gold_few, epoch_predict_few, top_k)
    ndcg5_few = mean_ndcg_score(epoch_gold_few, epoch_predict_few, top_k)

    recall5_zero = mean_recall_k(epoch_gold_zero, epoch_predict_zero, top_k)
    ndcg5_zero = mean_ndcg_score(epoch_gold_zero, epoch_predict_zero, top_k)

    return {'recall': recall5,
            'ndcg': ndcg5,
            'recall_frequent': recall5_frequent,
            'ndcg_frequent': ndcg5_frequent,
            'recall_few': recall5_few,
            'ndcg_few': ndcg5_few,
            'recall_zero': recall5_zero,
            'ndcg_zero': ndcg5_zero}


if __name__ == "__main__":
    epoch_gold = np.array([[1, 1, 0, 0, 0]])
    epoch_predict = np.array([[0.68, 0.23, 0.58, 0, 0]])
    print(mean_recall_k(epoch_gold, epoch_predict, 5))
    print(mean_ndcg_score(epoch_gold, epoch_predict, 5))