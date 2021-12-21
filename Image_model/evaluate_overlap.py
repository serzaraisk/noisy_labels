import sys
from typing import Dict, List

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
import scipy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('evaluate')
warnings.filterwarnings("ignore")


def average_aggregated_score(df, preds=None, threshold=None, group_by_col='image_url', overlap=5, n_resamples=100, metric=accuracy_score, encode=False):
    """
    df: honeypots dataframe, expected to have "ground_truth" and "label" and group_by_col columns
    preds: model predictions dataframe, expected to have "ground_truth" and "label" and group_by_col columns
    threshold: if threshold argument != None, preds dataframe is expected to have "confidence" column,
               models prediction will be mixed in the sampled markups only if corresponding "confidence" value is >= threshold
    group_by_col: name of column over which labels will be grouped by for sampling
                  (unique id of objects, could be image url, text etc.)
    overlap: numer of tolokers markups needed for majority vote
    n_resamples: number of subsets over which averaging will be applied
    metric: chosen metric for classification problems; expected to have two arguments as input: true
            labels and tolokers labels, for multiple arguments functions (such as f1_score with multiclass labels)
            use functools.partial or custom wrapper
    encode: if labels need to be encoded pass True
    Returns: score averaged over n_subsets
    """
    
    def get_slices(df):               #helper function, allows fast access to list of labels and ground truth by the key from group_by_col
        slices = defaultdict(list)
        slices_gt = dict()
        for i in np.array(df[[group_by_col, 'label', 'ground_truth']]):
            slices[i[0]].append(i[1])
            slices_gt[i[0]] = i[2]
        return slices, slices_gt
    
    if preds is not None:
        if threshold is None:
            preds = {x[0]:{'label':x[1]} for x in np.array(preds[[group_by_col, 'label']])}
            overlap -= 1 # if models precitions always added, sample less by 1
        else:
            preds = {x[0]:{'label':x[1], 'confidence':x[2]} for x in np.array(preds[[group_by_col, 'label', 'confidence']])}
            

    cnt_markups = df.groupby(group_by_col, as_index=False).count()
    enough_cnts = cnt_markups[cnt_markups['label'] >= overlap]
    
    mask = df[group_by_col].isin(set(enough_cnts[group_by_col]))
    samplable = df.loc[mask][[group_by_col, 'ground_truth', 'label']]     #enough to sample
    unsamplable = df.loc[~mask][[group_by_col, 'ground_truth', 'label']]  #less than overlap markups
    if encode:
        enc = preprocessing.LabelEncoder()
        all_lbl_values = list(np.unique(np.array(df[['ground_truth', 'label']])))
        if preds is not None:
            all_lbl_values.extend(list(np.unique(np.array(preds['label']))))
        enc.fit(all_lbl_values)

    scores = []
    for i in range(n_resamples):

        if len(samplable) == 0:
            sampled = unsamplable

        elif (len(samplable) != 0) and (len(unsamplable) != 0):
            sampled = samplable.groupby(group_by_col, as_index=False).sample(overlap)
            sampled = pd.concat([sampled, unsamplable])
        elif (len(samplable) != 0) and (len(unsamplable) == 0):
            sampled = samplable.groupby(group_by_col, as_index=False).sample(overlap)

        true = []
        agg = []
        slices, slices_gt = get_slices(sampled)
        
        for key in slices.keys():
            
            lbls = slices[key]
            if preds is not None:
                if threshold is not None:
                    pred, conf = preds[key]['label'], preds[key]['confidence']
                    if conf >= threshold:
                        if len(lbls) < overlap:
                            lbls.append(pred)
                        else:
                            lbls[0] = pred
                else:
                    pred = preds[key]['label']
                    lbls.append(pred)
            agg.append(scipy.stats.mode(lbls)[0][0])
            true.append(slices_gt[key])
        if encode:
            true = enc.transform(true)
            agg = enc.transform(agg)
        scores.append(metric(true, agg))
    
    return np.mean(scores), np.std(scores)


def build_conf_matrix(model_predictions, source):
    y_true = model_predictions['ground_truth']
    y_pred = model_predictions['label']

    label_order = list(set(y_true).union(set(y_pred)))  # keep track of only ground_truth labels
    classes = list(set(y_true))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    model_matrix = {
        'classes': classes,  # does not include ERROR classses (e.g. 404, invalid, etc.)
        'order': label_order,  # order of rows in confusion matrix
        'confusion_matrix': conf_matrix.tolist(),
    }

    logger.info('[{}] Per class stats:'.format(source))
    for clz in classes:
        idx = label_order.index(clz)
        recall = conf_matrix[idx][idx] / np.sum(conf_matrix[idx])
        precision = conf_matrix[idx][idx] / np.sum(conf_matrix[:, idx])
        logger.info(f'\t[{clz}], precision={precision * 100:.2f}% , recall={recall * 100:.2f}%')

    return model_matrix


def get_worker_skills(honeypot_df, smooth_factor: int):
    skills_dyn = dict()
    for performer, df in honeypot_df.groupby('workerId'):
        skills_dyn[performer] = (smooth_factor + np.sum(df['label'] == df['ground_truth'])) / (
                    2 * smooth_factor + len(df))
    return skills_dyn


def dyn_overlap_fixed_cnt(task_data,
                          required_cnt,
                          min_cnt=1,
                          max_cnt=100,
                          **kwargs):
    """
    Takes: task data dataframe, required_cnt for the number of "exactly the same" answers from different tolokers
    It is expected the order of rows (tasks) are sorted by submit_time, where the model predictions will be ALWAYS
    the first ones.
    Returns: number of resulting tolokers needed
    """
    class_cnts = defaultdict(lambda: 0)
    n_tolokers = len(task_data)

    overlap = np.nan
    for r in range(n_tolokers):
        cclass = task_data.iloc[r]['label']

        if r + 1 >= max_cnt:
            overlap = r + 1
            break

        class_cnts[cclass] += 1
        if class_cnts[cclass] == required_cnt:
            if r + 1 >= min_cnt:
                overlap = r + 1
                break
    return overlap, max(class_cnts, key=class_cnts.get)


def get_cnts_overlap(data_df: pd.DataFrame,
                     name: str,
                     dyn_overlap_config: dict,
                     overlap_keys: List[str] = None,
                     plot=False):
    cnts_overlap = {}
    pclasses = {}

    for task, df in tqdm(data_df.groupby('taskId')):
        cnt, pclass = dyn_overlap_fixed_cnt(df, **dyn_overlap_config)
        cnts_overlap[task] = cnt
        pclasses[task] = pclass

    if plot:
        if not overlap_keys:
            vals, cnts = np.unique(list(cnts_overlap.values()), return_counts=True)
        else:
            common_keys = np.intersect1d(list(cnts_overlap.keys()), overlap_keys)
            vals, cnts = np.unique([cnts_overlap[k] for k in common_keys], return_counts=True)
        average_overlap = sum(vals[cnts > 1] * cnts[cnts > 1]) / sum(cnts[cnts > 1])
        logger.info('Average [%s] overlap %.3f', name, average_overlap)
        sns.barplot(vals, cnts)
        plt.show()

    return cnts_overlap, pclasses


def dyn_overlap(task_data: pd.DataFrame,
                classes: np.array,
                skills_dyn: Dict[str, float],
                default_skill_value: float = 0.5):
    """
    Takes: task data dataframe
    Returns: epsilons of the most probable class for every round
    """
    n_classes = len(classes)
    # task_data = task_data.sort_values(by='assignment_submit_time')
    n_tolokers = len(task_data)
    score_matrix = np.zeros([n_tolokers, n_classes])
    for r in range(n_tolokers):
        cskill = skills_dyn.get(task_data.iloc[r]['workerId'], default_skill_value)
        score_matrix[r, :] = (1 - cskill) / (n_classes - 1)
        score_matrix[r, task_data.iloc[r]['label'] == classes] = cskill
    cum_scores = np.cumsum(np.log(score_matrix), axis=0)
    # now calculate eps for the class selected at each round
    z_hat = np.argmax(cum_scores, axis=1)
    sum_scores = np.sum(np.exp(cum_scores), axis=1)
    eps = np.exp(cum_scores[np.arange(n_tolokers), z_hat]) / sum_scores
    return eps


def compute_epsilons(labeled_df, skills_dyn, classes, name: str, default_skill_value: float, plot=False):
    epsilons = []
    for task, df in tqdm(labeled_df.groupby('taskId')):
        task_eps = dyn_overlap(df, classes, skills_dyn, default_skill_value)
        epsilons.append(task_eps)

    if plot:
        lengths = []
        for eps in epsilons:
            lengths.append(len(eps))

        vals, cnts = np.unique(lengths, return_counts=True)
        average_overlap = sum(vals[cnts > 1] * cnts[cnts > 1]) / sum(cnts[cnts > 1])
        logger.info(f'Average [%s] overlap %.2f', name, average_overlap)
        sns.barplot(vals, cnts)
        plt.show()

    return epsilons


def main(honeypots_json, model_predictions_json, configs):
    stats = {}
    K = configs['smoothing_factor']
    default_skill_value = configs['default_skill_value']
    make_plots = configs['plot']

    honeypots_df = pd.DataFrame.from_dict(honeypots_json, orient='columns')
    predictions_df = pd.DataFrame.from_dict(model_predictions_json, orient='columns')
    # annotated_df = pd.DataFrame.from_dict(annotated_json, orient='columns')

    # annotated_df = annotated_df[['label', 'workerId', 'taskId']]
    honeypots_df = honeypots_df[['ground_truth', 'label', 'workerId', 'taskId']]  # honeypots_df['workerId'] = 1
    predictions_df = predictions_df[['label', 'taskId', 'workerId', 'ground_truth']]  # predictions_df['workerId'] = 1

    unique_task_ids = set(pd.unique(honeypots_df['taskId']))
    logger.info('There are %d unique tasks in honeypots', len(unique_task_ids))

    # W/O smoothing skills ==================
    nonsmoothed_skills = get_worker_skills(honeypots_df, 0)
    average_worker_skill = np.mean(list(nonsmoothed_skills.values()))
    worker_honeypot_accuracy = accuracy_score(honeypots_df['ground_truth'], honeypots_df['label'])
    average_aggregated_accuracy = average_aggregated_score(honeypots_df, group_by_col='taskId')
    average_aggregated_accuracy_mixture = average_aggregated_score(honeypots_df, preds=predictions_df, group_by_col='taskId')

    logger.info('Workers average  skill: %.5f', average_worker_skill)
    logger.info('Workers honeypots accuracy: %.5f', worker_honeypot_accuracy)
    stats['average_worker_skill'] = average_worker_skill
    stats['honeypot_worker_accuracy'] = worker_honeypot_accuracy

    logger.info('Workers average aggregated accuracy on honeypots: %.5f', average_aggregated_accuracy)
    logger.info('Average aggregated accuracy on honeypots with mixture of models predictions and workers markups: %.5f', average_aggregated_accuracy_mixture)
    #  ==================

    # K-Smoothing skills  ================== # todo: non-golden is an eval subset of fasttext_classifier
    # classes = np.intersect1d(annotated_df.label, honeypots_df.label)
    classes = np.intersect1d(honeypots_df.label,
                             predictions_df.label)  # todo: (!) what if model will predict a class not present here (it is possible)?
    toloker_matrix = build_conf_matrix(honeypots_df, 'toloker')

    skills_dyn = get_worker_skills(honeypots_df, K)
    if make_plots:
        worker_epsilons = compute_epsilons(honeypots_df,
                                           skills_dyn,
                                           classes,
                                           'WORKERS',
                                           default_skill_value,
                                           plot=make_plots)  # todo: on non-golden
    cnts_overlap, pclasses = get_cnts_overlap(honeypots_df,
                                              'overlap WORKERS',
                                              configs,
                                              plot=make_plots)  # todo: on non-golden
    # =====================

    # AutoHelper =====================
    autohelper_accuracy = accuracy_score(predictions_df['ground_truth'],
                                         predictions_df['label'])  # todo: on golden dataset
    autohelper_name = predictions_df['workerId'].loc[0]
    skills_dyn[autohelper_name] = autohelper_accuracy
    logger.info('Autohelper accuracy: %.5f', autohelper_accuracy)
    stats['autohelper_accuracy'] = autohelper_accuracy

    # order matters a lot! as only first N appearances will be taken (bcs no presence of submission_time column)
    with_ah_honeypots = pd.concat([predictions_df, honeypots_df])
    if make_plots:
        epsilons_with_ah = compute_epsilons(with_ah_honeypots,
                                            skills_dyn,
                                            classes,
                                            'AH',
                                            default_skill_value,
                                            plot=make_plots)  # todo: on non-golden
    cnts_overlap_wa, pclasses_wa = get_cnts_overlap(with_ah_honeypots,
                                                    'overlap AH',
                                                    configs,
                                                    plot=make_plots,
                                                    overlap_keys=list(cnts_overlap.keys()))  # todo: on non-golden
    # =====================

    predictions_df['AggPred'] = predictions_df['taskId'].map(pclasses)  # todo: pclasses_golden
    predictions_df['AggPredWAH'] = predictions_df['taskId'].map(
        pclasses_wa)  # todo: pclasses_wa_golden  # переписать оценку точности через многократное сэмплирование

    accuracy_without = accuracy_score(predictions_df['ground_truth'], predictions_df['AggPred'])
    accuracy_with_ah = accuracy_score(predictions_df['ground_truth'], predictions_df['AggPredWAH'])
    logger.info("Aggregated accuracy without AutoHelpers is %.5f", accuracy_without)
    logger.info("Aggregated accuracy with AutoHelpers is %.5f", accuracy_with_ah)
    stats['aggregated_accuracy_without'] = accuracy_without
    stats['aggregated_accuracy_with_ah'] = accuracy_with_ah

    matrix = build_conf_matrix(predictions_df, 'autohelper')
    stats['matrix'] = matrix

    return stats


if __name__ == '__main__':
    args = sys.argv[1:]
    configs = {
        'default_skill_value': 0.5,
        'smoothing_factor': 10,
        'required_cnt': 3,
        'min_cnt': 3,
        'max_cnt': 5,
        'plot': False
    }  # todo: configs as param

    input_honeypot_pool_path = args[0]
    input_predictions_path = args[1]
    output_stats_path = args[2]

    with open(input_honeypot_pool_path, 'r') as f, open(input_predictions_path, 'r') as g:
        honeypots = json.load(f)
        predictions = json.load(g)
    stats = main(honeypots, predictions, configs)
    with open(output_stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
