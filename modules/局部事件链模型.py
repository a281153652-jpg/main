"""独立返回链探索，沿用 v4 观测尺度，不改变 v1 或原始候选标签。"""
from .局部事件模型 import LocalEventConfig, _fit_local_events

EVENT_MODEL_VERSION = 'local-piecewise-return-chain-v2'


def fit_local_return_chains(cycle, features, config=None):
    """同方向多段返回的局部证据；不把提案或支持次数解释为物理状态真值。"""
    result = _fit_local_events(cycle, features, config, merge_return_segments=True)
    result.update(event_model_version=EVENT_MODEL_VERSION,
                  event_return_chain_model='first_compatible_observed_post_reference',
                  event_reversal_guard='cumulative_observed_window_median_motion',
                  event_return_chain_changes_original_state=False)
    return result
