"""从整批动态特征学习形态分布；聚类概率不等于物理状态概率。"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from modules.放电状态识别 import MODEL_VERSION, STATES


BATCH_VERSION = 'batch-morphology-v1'
RESOLVED = ('open_circuit', 'effective_discharge', 'short_circuit')
UNDETERMINED = 'undetermined'
REQUIRED = (
    'cycle_id', 'model_version', 'data_quality', 'state', 'classification_reason',
    'sample_count', 'noise_current', 'activity_snr', 'robust_I_pp',
    'raw_event_fraction', 'oscillation_index', 'oscillation_support_fraction',
    'bounded_excursion_span_fraction', 'paired_response_fraction',
    'max_event_width_samples', 'plateau_fraction', 'post_stability_ratio',
    'transition_fraction', 'step_bic_gain', 'voltage_response_snr',
    'voltage_collapse_ratio',
)


@dataclass(frozen=True)
class BatchConfig:
    min_cycles: int = 32
    max_components: int = 8
    samples_per_component: int = 20
    max_fit_cycles: int = 20000
    membership_threshold: float = 0.8
    outlier_quantile: float = 0.999
    random_state: int = 42

    def __post_init__(self):
        if min(self.min_cycles, self.samples_per_component) < 2:
            raise ValueError('最少周期数及每组支持样本数必须至少为 2')
        if self.max_components < 1 or self.max_fit_cycles < self.min_cycles:
            raise ValueError('分组上限或拟合样本上限不合法')
        if not 0.5 < self.membership_threshold < 1 or not 0.5 < self.outlier_quantile < 1:
            raise ValueError('归属阈值和离群分位必须位于 (0.5, 1)')


def _feature_frame(frame):
    """使用无量纲动态特征；不将未测定主电平扣作零点。"""
    noise = pd.to_numeric(frame['noise_current'], errors='coerce').where(lambda v: v > 0)
    count = pd.to_numeric(frame['sample_count'], errors='coerce').where(lambda v: v > 0)
    values = pd.DataFrame(index=frame.index)
    for name in ('activity_snr', 'voltage_response_snr'):
        values['log_' + name] = np.log1p(pd.to_numeric(frame[name], errors='coerce').clip(lower=0))
    values['log_current_span_snr'] = np.log1p(frame['robust_I_pp'] / noise)
    values['max_event_fraction'] = frame['max_event_width_samples'] / count
    values['log_step_gain_per_point'] = np.log1p((frame['step_bic_gain'] / count).clip(lower=0))
    for name in ('raw_event_fraction', 'oscillation_index', 'oscillation_support_fraction',
                 'bounded_excursion_span_fraction', 'paired_response_fraction',
                 'plateau_fraction', 'post_stability_ratio', 'transition_fraction'):
        values[name] = pd.to_numeric(frame[name], errors='coerce')
    values['voltage_collapse_asinh'] = np.arcsinh(frame['voltage_collapse_ratio'])
    return values.replace([np.inf, -np.inf], np.nan)


def _prepare(values):
    # 平台不足引起的缺失值用批次中位数填充，并显式保留缺失指示列。
    columns = values.columns[values.notna().any()].tolist()
    values = values[columns]
    median = values.median()
    missing_columns = [c for c in columns if values[c].isna().any()]
    filled = values.fillna(median)
    augmented = filled.copy()
    for name in missing_columns:
        augmented[name + '__missing'] = values[name].isna().astype(float)
    center = augmented.median()
    iqr = augmented.quantile(.75) - augmented.quantile(.25)
    spread = augmented.std(ddof=0)
    # IQR=0 时使用整批标准差，不放大几乎常值列的舍入差异。
    floor = 1e-9 * np.maximum(1, np.abs(center))
    scale = iqr.where(iqr > floor, spread)
    usable = scale > floor
    features = augmented.columns[usable].tolist()
    x = ((augmented[features] - center[features]) / scale[features]).to_numpy(dtype=float)
    preprocessing = {
        'input_features': columns, 'missing_indicator_features': missing_columns,
        'impute_medians': median.to_dict(), 'retained_features': features,
        'centers': center[features].tolist(), 'scales': scale[features].tolist(),
    }
    return x, preprocessing


def _fit(x, config):
    # 按周期编号排序后均匀取拟合样本，保持可重复，并限制大批次成本。
    selected = np.linspace(0, len(x) - 1, min(len(x), config.max_fit_cycles), dtype=int)
    fit_x = x[selected]
    upper = min(config.max_components, max(1, len(fit_x) // config.samples_per_component),
                len(np.unique(fit_x, axis=0)))
    candidates, models = [], []
    with threadpool_limits(limits=1):
        for groups in range(1, upper + 1):
            model = GaussianMixture(n_components=groups, covariance_type='full',
                                    reg_covar=.05, n_init=3, max_iter=300,
                                    random_state=config.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                model.fit(fit_x)
            bic = float(model.bic(fit_x))
            counts = np.bincount(model.predict(fit_x), minlength=groups)
            supported = int(counts.min()) >= config.samples_per_component
            candidates.append({'groups': groups, 'bic': bic, 'converged': bool(model.converged_),
                               'min_component_samples': int(counts.min()), 'supported': supported})
            if model.converged_ and np.isfinite(bic) and supported:
                models.append((bic, model))
        if not models:
            return None, candidates, len(fit_x)
        return min(models, key=lambda entry: entry[0])[1], candidates, len(fit_x)


def learn_batch_states(results, output_dir, config=None, source_info=None):
    """无需人工逐周期标注即可拟合本批次；物理命名仍由动态证据支持。"""
    config = config or BatchConfig()
    missing = set(REQUIRED) - set(results.columns)
    if missing:
        raise ValueError(f'批次学习缺少特征列，需运行当前主程序重新提取: {sorted(missing)}')
    if results['cycle_id'].duplicated().any():
        raise ValueError('批次内 cycle_id 必须唯一')
    if not results['model_version'].eq(MODEL_VERSION).all():
        raise ValueError('仅支持当前 v3 原始特征；旧 v2 特征必须重新提取')
    if not results['state'].isin(STATES).all():
        raise ValueError('包含未支持的原始状态')
    original_order = results.index
    work = results.copy().reset_index(drop=True)
    work['batch_model_version'] = BATCH_VERSION
    work['morphology_cluster'] = -1
    work['cluster_membership'] = np.nan
    work['cluster_distance_squared'] = np.nan
    work['distribution_outlier'] = False
    work['automatic_state'] = work['state'].where(work['state'].isin(RESOLVED), UNDETERMINED)
    work['decision_status'] = np.where(work['automatic_state'].eq(UNDETERMINED),
                                       'undetermined', 'physical_evidence_candidate')
    work['review_priority'] = np.where(work['automatic_state'].eq(UNDETERMINED), 'high', 'sample')
    work['batch_reason'] = np.where(work['automatic_state'].eq(UNDETERMINED),
                                   'insufficient_physical_evidence', 'dynamic_evidence_only')
    valid = work['data_quality'].eq('ok')
    # 特征质量必须可判定；物理坏数据不参与学习，不以补零制造一个群。
    valid &= (np.isfinite(work['noise_current']) & work['noise_current'].gt(0)
              & np.isfinite(work['sample_count']) & work['sample_count'].gt(0)
              & np.isfinite(work['activity_snr']) & np.isfinite(work['robust_I_pp']))
    work.loc[~valid, ['automatic_state', 'decision_status', 'batch_reason', 'review_priority']] = [
        UNDETERMINED, 'undetermined', 'invalid_features_or_data', 'high']
    order = work.loc[valid].sort_values('cycle_id').index
    metadata = {
        'batch_model_version': BATCH_VERSION, 'feature_model_version': MODEL_VERSION,
        'learning_method': 'robust_scaling_full_gaussian_mixture_bic',
        'config': asdict(config), 'valid_cycles': int(valid.sum()),
        'uses_manual_labels': False, 'physical_accuracy_validated': False,
        'membership_is_physical_state_probability': False,
        'state_mapping': 'per_cycle_physical_evidence; clusters describe morphology only',
        'learning_status': 'insufficient_cycles',
        'source_info': source_info or {},
        'feature_values_sha256': hashlib.sha256(pd.util.hash_pandas_object(
            results.sort_values('cycle_id'), index=False).to_numpy().tobytes()).hexdigest(),
        'feature_values_hash_note': 'pandas 特征值哈希的 SHA-256；不是原始波形文件 SHA-256',
    }
    if len(order) >= config.min_cycles:
        x, preprocessing = _prepare(_feature_frame(work.loc[order]))
        metadata['preprocessing'] = preprocessing
        metadata['learning_status'] = 'constant_features' if not x.shape[1] else 'fit_failed'
        if x.shape[1]:
            model, candidates, fit_count = _fit(x, config)
            metadata.update({'bic_candidates': candidates, 'fit_cycles': fit_count})
            if model is not None:
                posterior = model.predict_proba(x)
                labels = posterior.argmax(axis=1)
                # 群组编号按中心排序固定，而非隐含对应四个物理状态。
                permutation = sorted(range(model.n_components), key=lambda i: tuple(model.means_[i]))
                remap = {old: new for new, old in enumerate(permutation)}
                clusters = np.array([remap[v] for v in labels], dtype=int)
                delta = x - model.means_[labels]
                distance = np.einsum('ni,nij,nj->n', delta, model.precisions_[labels], delta)
                limit = float(chi2.ppf(config.outlier_quantile, x.shape[1]))
                atypical = distance > limit
                membership = posterior.max(axis=1)
                work.loc[order, 'morphology_cluster'] = clusters
                work.loc[order, 'cluster_membership'] = membership
                work.loc[order, 'cluster_distance_squared'] = distance
                work.loc[order, 'distribution_outlier'] = atypical
                supported = work.loc[order, 'automatic_state'].ne(UNDETERMINED).to_numpy()
                typical = supported & ~atypical & (membership >= config.membership_threshold)
                work.loc[order[typical], 'decision_status'] = 'batch_supported_candidate'
                work.loc[order[typical], 'batch_reason'] = 'dynamic_evidence_and_typical_morphology'
                boundary = supported & ~atypical & ~typical
                work.loc[order[boundary], 'review_priority'] = 'medium'
                work.loc[order[boundary], 'batch_reason'] = 'overlapping_morphology_groups'
                # 离群不等于异常加工；仅对其拒绝给出确定自动状态。
                work.loc[order[atypical], ['automatic_state', 'decision_status', 'batch_reason',
                                          'review_priority']] = [UNDETERMINED, 'undetermined',
                                                                 'distribution_outlier', 'high']
                metadata.update({
                    'learning_status': 'learned', 'selected_groups': model.n_components,
                    'distance_limit_squared': limit,
                    'mixture': {'covariance_type': 'full', 'component_order': permutation,
                                'weights': model.weights_.tolist(), 'means': model.means_.tolist(),
                                'covariances': model.covariances_.tolist(),
                                'reg_covar': model.reg_covar},
                })
    work['batch_learning_status'] = metadata['learning_status']
    if 'review_required' in work:
        work['dynamic_review_required'] = work['review_required']
    # 这里是抽查调度标记，不是物理准确性已经验证的声明。
    work['review_required'] = work['review_priority'].eq('high')
    work['automatic_label_validated'] = False
    groups, representatives = [], []
    for cluster, subset in work.loc[work['morphology_cluster'].ge(0)].groupby('morphology_cluster'):
        counts = subset['automatic_state'].value_counts().to_dict()
        groups.append({'morphology_cluster': int(cluster), 'cycles': len(subset),
                       'mean_membership': float(subset['cluster_membership'].mean()),
                       'distribution_outliers': int(subset['distribution_outlier'].sum()),
                       'median_activity_snr': float(subset['activity_snr'].median()),
                       'median_raw_event_fraction': float(subset['raw_event_fraction'].median()),
                       'median_bounded_event_fraction': float(subset['bounded_excursion_span_fraction'].median()),
                       **{state: int(counts.get(state, 0)) for state in (*STATES, UNDETERMINED)}})
        ranked = subset.sort_values(['cluster_distance_squared', 'cycle_id'])
        for row, reason in ((ranked.iloc[0], 'closest_to_cluster_center'),
                            (ranked.iloc[-1], 'furthest_from_cluster_center')):
            representatives.append({'morphology_cluster': int(cluster), 'cycle_id': row['cycle_id'],
                                    'automatic_state': row['automatic_state'], 'selection_reason': reason,
                                    'review_priority': row['review_priority'],
                                    'cluster_membership': row['cluster_membership']})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    work.to_csv(output / '自学习状态结果.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(groups, columns=['morphology_cluster', 'cycles', 'mean_membership',
                                 'distribution_outliers', 'median_activity_snr', 'median_raw_event_fraction',
                                 'median_bounded_event_fraction', *STATES, UNDETERMINED]).to_csv(
        output / '波形群组统计.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(representatives, columns=['morphology_cluster', 'cycle_id', 'automatic_state',
                                         'selection_reason', 'review_priority', 'cluster_membership']).to_csv(
        output / '群组代表周期.csv', index=False, encoding='utf-8-sig')
    stats = work['automatic_state'].value_counts().reindex([*STATES, UNDETERMINED], fill_value=0)
    stats.rename_axis('automatic_state').rename('cycles').to_csv(output / '自动判别统计.csv', encoding='utf-8-sig')
    (output / '自学习模型.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + '\n', encoding='utf-8')
    print('\n整批特征自学习:', metadata['learning_status'], '| 形态群组:', metadata.get('selected_groups', 0))
    print('自动状态建议（undetermined 表示证据不足，不计为异常加工）:')
    print(stats.to_string())
    work.index = original_order
    return work
