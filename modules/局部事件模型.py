"""独立局部事件证据探索：不替换 v4 状态或把结构分数当作状态概率。"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter

EVENT_MODEL_VERSION = 'local-piecewise-event-v1'


@dataclass(frozen=True)
class LocalEventConfig:
    noise_multiplier: float = 4.0
    min_segment_samples: int = 32
    max_segments: int = 6
    max_boundary_candidates: int = 256

    def __post_init__(self):
        if isinstance(self.noise_multiplier, bool) or not np.isfinite(self.noise_multiplier) or self.noise_multiplier <= 0:
            raise ValueError('noise_multiplier 必须为有限正数')
        for name in ('min_segment_samples', 'max_segments', 'max_boundary_candidates'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} 必须为整数')
        if self.min_segment_samples < 8 or self.max_segments < 3 or self.max_boundary_candidates < 16:
            raise ValueError('局部事件配置不合法')


def _statistics(values):
    # 居中避免较大 DC 偏置时平方和消减；坐标统计保持原样本索引。
    offset = float(np.mean(values))
    y = values - offset
    x = np.arange(len(values), dtype=float)
    stats = np.vstack([np.r_[0., np.cumsum(v)] for v in (y, y*y, x, x*x, x*y)])
    return stats, offset


def _fit(stats, starts, end):
    starts = np.asarray(starts, dtype=int)
    n = end - starts
    sy, sy2, sx, sx2, sxy = stats[:, end, None] - stats[:, starts]
    variance = np.maximum(sx2 - sx*sx/n, np.finfo(float).tiny)
    covariance = sxy - sx*sy/n
    slope = covariance / variance
    intercept = (sy - slope*sx) / n
    sse = np.maximum(sy2 - sy*sy/n - covariance*covariance/variance, 0.)
    return sse, slope, intercept


def _segments(current, sigma, config):
    n = len(current)
    width = config.min_segment_samples
    filtered = median_filter(current, size=5, mode='nearest')
    changes = np.flatnonzero(np.diff(filtered) != 0) + 1
    boundaries = np.unique(np.r_[0, n, changes, np.arange(0, n, width)])
    capped = len(boundaries) > config.max_boundary_candidates
    if capped:
        boundaries = boundaries[np.unique(np.rint(np.linspace(
            0, len(boundaries)-1, config.max_boundary_candidates)).astype(int))]
    stats, offset = _statistics(current)
    count = len(boundaries)
    maximum = min(config.max_segments, n // width)
    dp = np.full((maximum+1, count), np.inf)
    back = np.full((maximum+1, count), -1, dtype=int)
    dp[0, 0] = 0.
    for k in range(1, maximum+1):
        for j in range(1, count):
            end = boundaries[j]
            eligible = np.flatnonzero((boundaries[:j] <= end-width)
                                     & np.isfinite(dp[k-1, :j]))
            if not len(eligible):
                continue
            errors, _, _ = _fit(stats, boundaries[eligible], end)
            costs = dp[k-1, eligible] + errors
            best = int(np.argmin(costs))
            dp[k, j] = costs[best]
            back[k, j] = eligible[best]
    # 固定观测尺度的 Gaussian working score。BIC 假设并未验证，不是概率。
    scores = dp[1:, -1]/sigma**2 + (3*np.arange(1, maximum+1)-1)*np.log(n)
    k = int(np.argmin(scores) + 1)
    j = count-1
    parts = []
    for level in range(k, 0, -1):
        previous = int(back[level, j])
        start, end = int(boundaries[previous]), int(boundaries[j])
        errors, slope, intercept = _fit(stats, np.array([start]), end)
        a = float(intercept[0]+slope[0]*start+offset)
        b = float(intercept[0]+slope[0]*(end-1)+offset)
        parts.append({'start': start, 'end': end, 'delta': b-a,
                      'slope': float(slope[0]), 'sse': float(errors[0])})
        j = previous
    return list(reversed(parts)), stats, offset, float(scores[k-1]), capped, count


def _references(current, start, end, sigma, config, reverse=False):
    """实际窗口均有足够样本；局部背景只作形态参考，不校正测量零点。"""
    width = config.min_segment_samples
    if end-start < width:
        return []
    starts = range(end-width, start-1, -1) if reverse else range(start, end-width+1)
    result = []
    coordinate = np.arange(width, dtype=float)
    centered_x = coordinate - coordinate.mean()
    for first in starts:
        values = current[first:first+width]
        std = float(np.std(values))
        motion = abs(float(np.dot(values-values.mean(), centered_x)
                           / np.dot(centered_x, centered_x))) * (width-1)
        if std <= config.noise_multiplier*sigma and motion <= config.noise_multiplier*sigma:
            result.append({'start': first, 'end': first+width,
                           'median': float(np.median(values)), 'std': std, 'motion': motion})
    return result


def _structure_scores(current, first, last, parts, sigma):
    """同一观测区间比较常值、线性、单阶跃与提案分段结构。"""
    values = current[first:last]
    n = len(values)
    stats, _ = _statistics(values)
    constant_sse = float(np.sum((values-values.mean())**2))
    linear_sse = float(_fit(stats, np.array([0]), n)[0][0])
    s = np.cumsum(values-values.mean())
    square = np.cumsum((values-values.mean())**2)
    split = np.arange(1, n)
    step_errors = (square[split-1]-s[split-1]**2/split
                   + square[-1]-square[split-1]
                   - (s[-1]-s[split-1])**2/(n-split))
    step_sse = float(np.maximum(step_errors.min(), 0.))
    null_scores = np.array([constant_sse/sigma**2+np.log(n),
                            linear_sse/sigma**2+2*np.log(n),
                            step_sse/sigma**2+3*np.log(n)])
    cuts = sorted(set([first, last, *[p['start'] for p in parts if first < p['start'] < last],
                       *[p['end'] for p in parts if first < p['end'] < last]]))
    sse = sum(float(_fit(stats, np.array([a-first]), b-first)[0][0])
              for a, b in zip(cuts[:-1], cuts[1:]))
    model_score = sse/sigma**2+(3*(len(cuts)-1)-1)*np.log(n)
    which = int(np.argmin(null_scores))
    return float(null_scores[which]-model_score), ('constant', 'linear_drift', 'single_step')[which]


def _drive_reference(voltage, start, end, sigma, config):
    """预事件内选常值优于趋势/阶跃的实际窗口，并保留驱动纹波尺度。"""
    width = config.min_segment_samples
    x = np.arange(width, dtype=float)-(width-1)/2
    x2 = float(np.dot(x, x))
    options = []
    for first in range(start, end-width+1):
        values = voltage[first:first+width]
        centered = values-values.mean()
        constant_sse = float(np.dot(centered, centered))
        slope = float(np.dot(centered, x)/x2)
        linear_sse = max(0., constant_sse-slope*slope*x2)
        sums = np.cumsum(centered)
        squares = np.cumsum(centered**2)
        splits = np.arange(1, width)
        step_errors = (squares[splits-1]-sums[splits-1]**2/splits
                       + squares[-1]-squares[splits-1]
                       - (sums[-1]-sums[splits-1])**2/(width-splits))
        step_sse = max(0., float(step_errors.min()))
        # 候选模型共享一个局部尺度。驱动纹波不等于二阶差分测量噪声。
        scale = max(sigma, float(np.std(values)))
        if ((constant_sse-linear_sse)/scale**2 <= np.log(width)
                and (constant_sse-step_sse)/scale**2 <= 2*np.log(width)):
            options.append({'start': first, 'end': first+width,
                            'median': float(np.median(values)), 'std': float(np.std(values)),
                            'scale': scale})
    return max(options, key=lambda ref: (ref['median'], -ref['start'])) if options else None


def fit_local_events(cycle, features, config=None):
    """返回 event_* 独立证据；只检测局部 rise/return，不给物理状态真值。"""
    return _fit_local_events(cycle, features, config, merge_return_segments=False)


def _resolved_path_reversal(current, start, end, direction, limit, width, window_medians=None):
    """v2 路径守卫：量化小台阶累计反向运动也不能连接成一条腿。"""
    if end-start < width:
        return False
    medians = (window_medians[start:end-width+1] if window_medians is not None else
               np.array([np.median(current[first:first+width])
                         for first in range(start, end-width+1)]))
    oriented = direction*medians
    return bool(np.any(np.maximum.accumulate(oriented)-oriented > limit))


def _background_has_opposite_ownership(references, left, right, orientation, limit):
    """兼容外层背景包围两个反向高峰时，峰间谷的事件归属存在歧义。"""
    before = [ref['median'] for ref in references if ref['end'] <= left['start']
              and orientation*(left['median']-ref['median']) < -limit]
    after = sorted(ref['median'] for ref in references if ref['start'] >= right['end']
                   and orientation*(right['median']-ref['median']) < -limit)
    if not before or not after:
        return False
    after = np.asarray(after)
    for median in before:
        position = int(np.searchsorted(after, median-2*limit))
        if position < len(after) and after[position] <= median+2*limit:
            return True
    return False


def _fit_local_events(cycle, features, config=None, *, merge_return_segments=False):
    """共享计算；v1 保持原行为，返回连接及累计反向守卫只由独立 v2 启用。"""
    config = config or LocalEventConfig()
    reliable = features.get('current_scale_reliable', False)
    result = {'event_model_version': EVENT_MODEL_VERSION, 'event_supported': False,
              'event_reason': 'no_resolved_local_rise_return', 'event_candidate_count': 0,
              'event_supported_count': 0, 'event_boundary_truncated': False,
              'event_scale_reliable': (isinstance(reliable, (bool, np.bool_)) and bool(reliable)),
              'event_config_noise_multiplier': config.noise_multiplier,
              'event_config_min_segment_samples': config.min_segment_samples,
              'event_structure_score_is_probability': False,
              'event_baseline_model': 'compatible_local_reference_intervals',
              'event_scale_assumption': 'base_v4_conservative_current_scale',
              'event_supported_count_is_proposal_count': True,
              'event_voltage_reference_model': 'constant_vs_linear_and_single_step',
              'event_boundary_flag_is_proven_truncation': False}
    required = ['time', 'voltage', 'current']
    if not all(name in cycle for name in required):
        raise ValueError('局部事件输入需要 time/voltage/current')
    if features.get('data_quality') != 'ok':
        result['event_reason'] = 'invalid_base_data_quality'
        return result
    time = cycle['time'].to_numpy(dtype=float)
    if not np.isfinite(time).all() or len(time) < 2 or np.any(np.diff(time) <= 0):
        result['event_reason'] = 'invalid_local_timebase'
        return result
    intervals = np.diff(time)
    if np.max(np.abs(intervals/np.median(intervals)-1)) > .01:
        result['event_reason'] = 'invalid_local_timebase'
        return result
    if features.get('sample_count') != len(cycle):
        result['event_reason'] = 'base_feature_sample_count_mismatch'
        return result
    sigma = float(features.get('noise_current', np.nan))
    sigma_u = float(features.get('noise_voltage', np.nan))
    if not result['event_scale_reliable'] or not np.isfinite(sigma) or sigma <= 0:
        result['event_reason'] = 'untrusted_current_scale'
        return result
    offset = float(features.get('current_zero_offset_A', 0.))
    if not np.isfinite(offset):
        result['event_reason'] = 'invalid_current_zero_offset'
        return result
    current = cycle['current'].to_numpy(dtype=float) - offset
    voltage = np.abs(cycle['voltage'].to_numpy(dtype=float))
    if (len(current) < 3*config.min_segment_samples or not np.isfinite(current).all()
            or not np.isfinite(voltage).all() or not np.isfinite(sigma_u) or sigma_u <= 0):
        result['event_reason'] = 'insufficient_or_nonfinite_samples_or_scale'
        return result
    parts, _, _, score, capped, boundary_count = _segments(current, sigma, config)
    result.update(event_selected_segments=len(parts), event_segment_cap_hit=len(parts)==config.max_segments,
                  event_boundary_candidates=boundary_count, event_boundary_grid_capped=capped,
                  event_segmentation_working_bic=score)
    limit = config.noise_multiplier*sigma
    window_medians = (np.median(np.lib.stride_tricks.sliding_window_view(
        current, config.min_segment_samples), axis=1) if merge_return_segments else None)
    ownership_references = None
    def resolved_reverse_boundary(index, orientation):
        boundary = parts[index]['start']
        width = config.min_segment_samples
        before = np.median(current[boundary-width:boundary])
        after = np.median(current[boundary:boundary+width])
        return orientation*(after-before) < -limit

    candidates = []
    for orientation in (1, -1):
        for falling in range(1, len(parts)):
            if orientation*parts[falling]['delta'] >= -limit:
                continue
            rising = falling-1
            # 量化爬升可先出现几个近常值台阶，不能只保留最后一个线性片段。
            # 保留最前面的参考片段；遇到已分辨的反向腿即停止，不连接独立事件。
            while rising > 1:
                previous = rising-1
                if (orientation*parts[previous]['delta'] < -limit
                        or resolved_reverse_boundary(previous, orientation)
                        or resolved_reverse_boundary(rising, orientation)):
                    # 反向跳变后的片段留作新事件的背景，而非吞入上升腿。
                    break
                rising -= 1
            rise_parts = parts[rising:falling]
            if not rise_parts or max(orientation*p['delta'] for p in rise_parts) <= limit:
                continue
            candidate = {'event_supported': False, 'event_reason': 'unresolved_local_peak_amplitude',
                         'event_orientation': orientation, 'event_start_index': parts[rising]['start'],
                         'event_end_index': parts[falling]['end']}
            candidates.append(candidate)
            if rising == 0 or abs(parts[rising-1]['delta']) > limit:
                candidate.update(event_reason='background_transition_or_boundary_context_required',
                                 event_boundary_truncated=True)
                continue
            pre_segment = parts[rising-1]
            after_start = parts[falling]['end']
            after_end = parts[falling+1]['end'] if falling+1 < len(parts) else after_start
            pre = _references(current, pre_segment['start'], pre_segment['end'], sigma, config, reverse=True)
            post = _references(current, after_start, after_end, sigma, config)
            compatible = next(((left, right) for right in post for left in pre
                               if abs(left['median']-right['median']) <= 2*limit), None)
            returning = falling
            if merge_return_segments:
                # 只连接同方向/未分辨的返回片段；首个兼容实际背景窗口结束搜索。
                # 累计反向运动捕获多次小网格台阶形成的另一脉冲，不能只逐边界判断。
                while True:
                    compatible = None
                    for right in post:
                        left = next((ref for ref in pre
                                     if abs(ref['median']-right['median']) <= 2*limit), None)
                        # 路径守卫不依赖前参考；每个 post 只检查一次，
                        # 避免长背景中大量兼容 pre 重复扫描同一条返回路径。
                        if left is not None and not _resolved_path_reversal(
                                current, parts[falling]['start'], right['end'],
                                -orientation, limit, config.min_segment_samples, window_medians):
                            compatible = (left, right)
                            break
                    if compatible is not None or returning+1 >= len(parts):
                        break
                    following = returning+1
                    boundary = parts[following]['start']
                    width = config.min_segment_samples
                    before = np.median(current[boundary-width:boundary])
                    after = np.median(current[boundary:boundary+width])
                    if (orientation*parts[following]['delta'] > limit
                            or orientation*(after-before) > limit
                            or _resolved_path_reversal(current, parts[falling]['start'],
                                                       parts[following]['end'],
                                                       -orientation, limit, width, window_medians)):
                        break
                    returning = following
                    after_start = parts[returning]['end']
                    after_end = parts[returning+1]['end'] if returning+1 < len(parts) else after_start
                    post = _references(current, after_start, after_end, sigma, config)
                candidate.update(event_end_index=parts[returning]['end'],
                                 event_return_chain_segments=returning-falling+1,
                                 event_return_chain_start_index=parts[falling]['start'],
                                 event_return_chain_end_index=parts[returning]['end'])
            if compatible is None:
                candidate.update(event_reason='background_transition_or_boundary_context_required',
                                 event_boundary_truncated=True)
                continue
            left, right = compatible
            first, last = parts[rising]['start'], parts[returning]['end']
            width = config.min_segment_samples
            medians = np.array([np.median(current[s:s+width]) for s in range(first, last-width+1)])
            if not len(medians):
                continue
            peak_start = first+int(np.argmax(orientation*medians))
            peak = float(np.median(current[peak_start:peak_start+width]))
            rise_amplitude = orientation*(peak-left['median'])
            return_amplitude = orientation*(peak-right['median'])
            candidate.update(event_peak_A=peak, event_peak_index=peak_start+width//2,
                             event_peak_reference_start_index=peak_start,
                             event_peak_reference_end_index=peak_start+width,
                             event_pre_reference_start_index=left['start'],
                             event_pre_reference_end_index=left['end'],
                             event_post_reference_start_index=right['start'],
                             event_post_reference_end_index=right['end'],
                             event_pre_current_A=left['median'], event_post_current_A=right['median'],
                             event_pre_std_A=left['std'], event_post_std_A=right['std'],
                             event_rise_samples=parts[falling]['start']-first,
                             event_fall_samples=last-parts[falling]['start'],
                             event_rise_amplitude_scale_ratio=rise_amplitude/sigma,
                             event_return_amplitude_scale_ratio=return_amplitude/sigma)
            # 实际峰窗口的幅度和拟合运动同时支持；一网格短尖峰不能靠外推获支持。
            if rise_amplitude <= limit or return_amplitude <= limit:
                continue
            if merge_return_segments and (
                    _resolved_path_reversal(current, first, peak_start+width,
                                            orientation, limit, width, window_medians)
                    or _resolved_path_reversal(current, peak_start, right['end'],
                                               -orientation, limit, width, window_medians)):
                candidate['event_reason'] = 'resolved_reversal_within_event_chain'
                continue
            if merge_return_segments:
                if ownership_references is None:
                    ownership_references = _references(current, 0, len(current), sigma, config)
                if _background_has_opposite_ownership(ownership_references, left, right,
                                                      orientation, limit):
                    candidate['event_reason'] = 'ambiguous_background_owned_by_opposite_excursions'
                    continue
            gain, null_model = _structure_scores(current, left['start'], right['end'], parts, sigma)
            # 电压可先于电流上升响应。驱动参考在同一事件之前的背景片段中独立查找，
            # 不以已降压的最后一个电流背景窗口充当原始驱动电压，也不借用事件后的变化。
            drive_ref = _drive_reference(voltage, pre_segment['start'], pre_segment['end'], sigma_u, config)
            if drive_ref is None:
                candidate['event_reason'] = 'no_stable_prior_voltage_drive_reference'
                continue
            u_ref = drive_ref['median']
            background_edge = max(left['median'], right['median']) if orientation > 0 else min(left['median'], right['median'])
            outside = orientation*(current[first:last]-background_edge) > limit
            response = u_ref-voltage[first:last] > config.noise_multiplier*drive_ref['scale']
            paired_count = int(np.count_nonzero(outside & response))
            candidate.update(event_bic_gain=gain, event_null_model=null_model,
                             event_voltage_reference_V=u_ref, event_joint_response_samples=paired_count,
                             event_voltage_reference_start_index=drive_ref['start'],
                             event_voltage_reference_end_index=drive_ref['end'],
                             event_voltage_reference_std_V=drive_ref['std'],
                             event_voltage_response_scale_V=drive_ref['scale'],
                             event_voltage_response_supported=paired_count >= width)
            if gain <= 0:
                candidate['event_reason'] = 'null_structure_model_preferred'
            elif paired_count < width:
                candidate['event_reason'] = 'insufficient_same_event_voltage_response'
            else:
                candidate.update(event_supported=True, event_reason='resolved_local_rise_return_with_joint_response')
    result['event_candidate_count'] = len(candidates)
    result['event_supported_count'] = sum(c['event_supported'] for c in candidates)
    result['event_candidate_rejection_reasons'] = '|'.join(sorted(set(
        c['event_reason'] for c in candidates if not c['event_supported'])))
    if candidates:
        best = max(candidates, key=lambda c: (c['event_supported'], c.get('event_bic_gain', -np.inf),
                                              min(c.get('event_rise_amplitude_scale_ratio', 0.),
                                                  c.get('event_return_amplitude_scale_ratio', 0.))))
        result.update(best)
    return result
