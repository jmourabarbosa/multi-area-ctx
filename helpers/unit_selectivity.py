from typing import Sequence
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def unit_selectivity(unit):
    """
    Test if a given unit has a prefered task

    Parameters
    ----------
    unit: dict

    Returns
    -------
    is_sel: bool
        Has a prefered task?
    pref_task: str
        Which task is the prefered one. 'none' if is_sel == False
    p: float
        p-value of the Mann Whitney test
    task_bsl: dict
        For each task, a np.ndarray with baselines
    """
    t = unit['time']
    is_bsl = t < 0
    # all pooled stimuli names (le_lc, ri_lc, hi_lc, lo_lc ...)
    
    #all_stims = [s for s in unit.keys() if '_' in s and len(s) < 6]
    
    all_stims = [s for s in unit.keys() if len(s.split("_")) == 3 and s[-1] != "o"]
    # all tasks names (lc, pc)
    tasks = set([s[-2:] for s in all_stims])
    # all pooled stimuli names split per task
    
    #stim_tasks = {t: [s for s in unit.keys() if t in s and len(s) < 6] for t in tasks}
    stim_tasks = {t: [s for s in unit.keys() if t in s and len(s.split("_")) == 3 and s[-1] != "o"] for t in tasks}

    # all neuronal responses pooled by task (lc/pc)
    task_resp = {t: np.vstack([unit[s] for s in stim_tasks[t]]) for t in tasks}
    # Average baseline activity per task
    task_bsl = {t: r[:, is_bsl].mean(1) for t, r in task_resp.items()}
    # Mann Whitney test
    u, p = mannwhitneyu(*task_bsl.values(), alternative='two-sided')
    is_sel = p < 5e-2
    if is_sel:
        # Which task has the highest baseline activity
        avg_bsl = {t: r.mean() for t, r in task_bsl.items()}
        pref_task = min(avg_bsl, key=avg_bsl.__getitem__)
    else:
        pref_task = 'none'
    return is_sel, pref_task, p, task_bsl


def get_select_units(units: Sequence, multi_method='fdr_bh'):
    """
    Check cells preferences for a task based on their baseline activity

    Parameters
    ----------
    units: list
    multi_method: str

    Returns
    -------
    selectivity
    prefered_task
    bsl
    """
    preferences = [unit_selectivity(u) for u in units]
    pvals = [p[2] for p in preferences]
    bsl = [p[3] for p in preferences]
    preferred_task = np.array([x[1] for x in preferences])
    if multi_method is not None:
        selectivity, c_pvals, *_ = multipletests(pvals, method=multi_method)
        preferred_task[np.logical_not(selectivity)] = 'none'
    else:
        selectivity = np.array([x[0] for x in preferences])

    return selectivity, preferred_task, bsl
