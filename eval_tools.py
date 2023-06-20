# -*- coding: utf-8 -*-
import evaluation_toolkit.source.toolkit.utils.misc as my_misc
from evaluation_toolkit.source.toolkit import settings
from evaluation_toolkit.source.toolkit.metrics.general_metrics import BadPix
from evaluation_toolkit.source.toolkit.evaluations import submission_evaluation
from evaluation_toolkit.source.toolkit.utils import file_io, log, misc
import scipy.ndimage.interpolation as sci

def compute_scores(scene, metrics, algo_result, with_visualization=False):
    scores = dict()

    # resolution for evaluation depends on metric
    low_res_metrics = scene.get_applicable_metrics_low_res(metrics)
    if low_res_metrics:
        scene.set_low_gt_scale()
        scores, vis_list = add_scores(low_res_metrics, scene, algo_result, scores, with_visualization)

    high_res_metrics = scene.get_applicable_metrics_high_res(metrics)
    if high_res_metrics:
        scene.set_high_gt_scale()
        scores, vis_list = add_scores(high_res_metrics, scene, algo_result, scores, with_visualization)

    if with_visualization:
        return scores, vis_list
    else:
        return scores

def compute_errorimage(scene, metrics, algo_result, with_visualization=False):
    scores = dict()

    # resolution for evaluation depends on metric
    low_res_metrics = scene.get_applicable_metrics_low_res(metrics)
    if low_res_metrics:
        scene.set_low_gt_scale()
        scores, vis_list = add_scores(low_res_metrics, scene, algo_result, scores, with_visualization)

    high_res_metrics = scene.get_applicable_metrics_high_res(metrics)
    if high_res_metrics:
        scene.set_low_gt_scale()
        scores, vis_list = add_scores(high_res_metrics, scene, algo_result, scores, with_visualization)

    if with_visualization:
        return scores, vis_list
    else:
        return scores

def add_scores(metrics, scene, algo_result, scores, with_visualization=False):
    gt = scene.get_gt()
    scaled_result = get_scaled_algo_result(algo_result, scene)
    #print scene
    vis_list = []
    for metric in metrics:
        if with_visualization:
            score, vis = metric.get_score(scaled_result, gt, scene, with_visualization=True)
            vis_list.append(vis)
        else:
            score = metric.get_score(scaled_result, gt, scene, with_visualization=False)

        metric_data = {"value": float(score)}

        #log.info("Score %5.2f for: %s, %s, Scale: %0.2f" %
        #         (score, metric.get_display_name(), scene.get_display_name(), scene.gt_scale))

        scores[metric.get_id()] = metric_data

    return scores, vis_list

def get_scaled_algo_result(algo_result, scene):
    if scene.gt_scale != 1:
        algo_result = sci.zoom(algo_result, scene.gt_scale, order=0)
    return algo_result