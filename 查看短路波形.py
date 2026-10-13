"""保留旧导入入口，统一使用动态特征模型的候选评分，避免误用旧阈值。"""
from modules.短路候选分析 import calculate_short_score, short_candidate_analysis

__all__ = ['calculate_short_score', 'short_candidate_analysis']
