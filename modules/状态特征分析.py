from pathlib import Path

import pandas as pd

FEATURES = [
    'Ipeak', 'Irms', 'I_pp', 'I_std', 'signed_I_pp', 'signed_I_std', 'robust_I_pp',
    'change_count', 'reversal_count', 'oscillation_index', 'oscillation_span_ratio',
    'noise_current', 'activity_snr', 'I_jump', 'step_bic_gain', 'transition_fraction',
    'plateau_fraction', 'post_current_std', 'post_stability_ratio',
    'voltage_drop_snr', 'voltage_collapse_ratio', 'voltage_std',
    'energy_J', 'energy_signed_J', 'discharge_ratio', 'short_score',
]


def analyze_features(result_file, output_dir):
    df = pd.read_csv(result_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    available = [feature for feature in FEATURES if feature in df]
    records = []
    for feature in available:
        values = df[feature]
        records.append({
            'feature': feature, 'valid_count': values.count(),
            'min': values.min(), '25%': values.quantile(.25), '50%': values.quantile(.5),
            '75%': values.quantile(.75), '90%': values.quantile(.9),
            '95%': values.quantile(.95), 'max': values.max(),
        })
    stats = pd.DataFrame(records, columns=['feature', 'valid_count', 'min', '25%', '50%',
                                         '75%', '90%', '95%', 'max'])
    stats.to_csv(output / '状态特征统计.csv', index=False, encoding='utf-8-sig')
    df.groupby('state')[available].mean().to_csv(output / '各状态特征均值.csv', encoding='utf-8-sig')
    print('\n各状态特征均值:')
    print(df.groupby('state')[available].mean())
    return stats
