# -*- coding: utf-8 -*-
import numpy as np


# ----------- this 0.07 is unchangeable ------#
ERROR_RANGE = 0.07

class Evaluator:
    # 17.11.19.1
    boundary_offset = 15

    def get_boundary_mask(self, array, ignore_boundary=True):
        assert array.ndim >= 2
        hei, wid = array.shape[0], array.shape[1]
        if ignore_boundary:
            mask = np.full(array.shape, fill_value=0, dtype=np.bool)
            f_offset = self.boundary_offset
            mask[f_offset:hei-f_offset, f_offset:wid-f_offset] = True
        else:
            mask = np.full(array.shape, fill_value=1, dtype=np.bool)
        return mask

    def error_acc(self, disp_pre, disp_gt, error_range=ERROR_RANGE, ignore_boundary=True, eval_mask=None):
        err_pre = (abs(disp_pre - disp_gt) <= error_range).astype(np.uint8)

        #acc_no_mask = np.mean(err_pre)
        mask = self.get_boundary_mask(disp_gt, ignore_boundary)
        if eval_mask is not None:
            mask *= eval_mask

        valid_err_pre = err_pre[mask]

        value_acc = np.mean(valid_err_pre)

        # boundary as unvalid
        err_pre[~mask] = 1
        return err_pre, value_acc  #, acc_no_mask

    def mse(self, algo_result, disp_gt, factor=100, ignore_boundary=True, eval_mask=None):
        mask = self.get_boundary_mask(disp_gt, ignore_boundary)
        if eval_mask is not None:
            mask *= eval_mask

        with np.errstate(invalid="ignore"):
            diff = np.square(disp_gt - algo_result)
            diff[~mask] = 0

        return diff, np.average(diff[mask]) * factor