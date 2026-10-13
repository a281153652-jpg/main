"""输出判别依据的逐项审计，不据此覆写状态或把特征现象当成物理标签。"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modules.放电状态识别 import DynamicConfig


def write_decision_audit(results, output_dir, config=None):
    config = config or DynamicConfig()
    def numeric(name):
        return pd.to_numeric(results.get(name, pd.Series(np.nan, index=results.index)), errors='coerce')

    columns = ['cycle_id', 'state', 'automatic_state', 'classification_reason', 'batch_reason',
               'morphology_cluster', 'current_noise_model', 'noise_current', 'noise_current_mad',
               'current_level_step_A', 'activity_snr', 'sample_count', 'raw_event_count',
               'max_event_width_samples', 'oscillation_support_fraction',
               'bounded_excursion_span_fraction', 'paired_response_fraction',
               'pre_sample_count', 'post_sample_count', 'plateau_supported']
    audit = results[[c for c in columns if c in results]].copy()
    n = numeric('sample_count').where(lambda s: s > 0)
    audit['raw_longest_event_fraction'] = numeric('max_event_width_samples') / n
    audit['activity_above_reference_scale'] = numeric('activity_snr') > config.noise_multiplier
    audit['bounded_event_span_supported'] = (numeric('bounded_excursion_span_fraction')
                                             >= config.oscillation_persistence)
    audit['voltage_overlap_supported'] = numeric('paired_response_fraction') >= config.oscillation_persistence
    audit['oscillation_reversal_count_supported'] = numeric('reversal_count') >= config.min_reversals
    audit['oscillation_window_coverage_supported'] = numeric('oscillation_index') >= config.oscillation_persistence
    audit['oscillation_span_supported'] = numeric('oscillation_span_ratio') >= config.oscillation_persistence
    audit['oscillation_time_support_supported'] = (numeric('oscillation_support_fraction')
                                                  >= config.oscillation_persistence)
    audit['long_raw_event_without_bounded_support'] = (
        audit['raw_longest_event_fraction'].ge(config.oscillation_persistence)
        & ~audit['bounded_event_span_supported'] & audit['voltage_overlap_supported'])
    reason = results['classification_reason']
    audit['resolution_limited_long_event'] = (reason.eq('resolution_limited_events')
                                             & audit['raw_longest_event_fraction'].ge(config.oscillation_persistence))
    audit['numerical_scale_requires_review'] = ~results.get(
        'noise_scale_reliable_for_learning', pd.Series(True, index=results.index))
    flags = ('long_raw_event_without_bounded_support', 'resolution_limited_long_event',
             'numerical_scale_requires_review')
    audit['diagnostic_flags'] = [';'.join(name for name in flags if bool(audit.at[index, name]))
                               for index in audit.index]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / '判别证据审计.csv', index=False, encoding='utf-8-sig')
    metadata = {
        'reference_dynamic_config': asdict(config),
        'reference_config_note': '仅按给定/默认 v3 参数审计；不会重新分类，不等于已发现误判。',
        'diagnostic_counts': {name: int(audit[name].sum()) for name in flags},
        'raw_waveforms_available_in_feature_table': False,
        'long_raw_event_note': '原始事件长而内部有界事件支持不足，可能涉及周期截断、基线选择或事件破碎；需原始样本核查。',
    }
    (output / '判别证据审计说明.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return audit
