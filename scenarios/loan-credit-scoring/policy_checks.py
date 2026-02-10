import yaml
import pandas as pd
from typing import List, Dict, Any


def evaluate_group_min_positive_rate(df: pd.DataFrame, target: str, dimension: str, threshold: float, age_bucket_method: str = None, age_buckets: int = 3) -> Dict[str, Any]:
    """Compute min positive-rate per group (by dimension) and evaluate against threshold.

    Returns a dict mimicking the .venturalitica metric object:
    {
      "control_id": str,
      "metric_key": "group_min_positive_rate",
      "threshold": threshold,
      "actual_value": min_rate,
      "operator": "gt",
      "passed": bool,
      "metadata": {"groups": {...}}
    }
    """
    # Prepare target as binary (positive = 1)
    y = df[target]
    # Try to coerce to numeric
    try:
        y_num = pd.to_numeric(y, errors='coerce')
    except Exception:
        y_num = y

    # Determine positive label: if values are (0,1) or (1,2) -> choose the higher numeric value as 'positive'
    unique_vals = sorted(pd.unique(y_num[~pd.isna(y_num)]))
    if len(unique_vals) == 2:
        positive_val = unique_vals[-1]
    else:
        # fallback: consider non-zero as positive
        positive_val = 1

    # Build groups
    if dimension not in df.columns:
        raise KeyError(f"Dimension column '{dimension}' not found in dataframe")

    series = df[dimension]
    if age_bucket_method == 'quantiles' and pd.api.types.is_numeric_dtype(series):
        groups = pd.qcut(series, q=age_buckets, duplicates='drop')
    else:
        groups = series

    grouped = df.groupby(groups)

    rates = {}
    for name, g in grouped:
        grp_y = g[target]
        try:
            grp_y_num = pd.to_numeric(grp_y, errors='coerce')
            pos_rate = (grp_y_num == positive_val).sum() / len(grp_y_num.dropna()) if len(grp_y_num.dropna())>0 else 0.0
        except Exception:
            pos_rate = float((grp_y == positive_val).sum() / len(grp_y))
        rates[str(name)] = pos_rate

    if len(rates) == 0:
        min_rate = 0.0
    else:
        min_rate = min(rates.values())

    passed = bool(min_rate >= threshold)

    return {
        "metric_key": "group_min_positive_rate",
        "actual_value": min_rate,
        "threshold": threshold,
        "operator": "gt",
        "passed": passed,
        "metadata": {"groups": rates}
    }


def evaluate_data_policy_controls(policy_path: str, df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    """Evaluate specialized data policy controls (currently supports 'group_min_positive_rate')."""
    doc = yaml.safe_load(open(policy_path))
    reqs = doc.get('system-security-plan', {}).get('control-implementation', {}).get('implemented-requirements', [])
    results = []
    for r in reqs:
        props = {p['name']: p['value'] for p in r.get('props', [])}
        metric = props.get('metric_key')
        if metric == 'group_min_positive_rate':
            dimension = props.get('input:dimension')
            thresh = float(props.get('threshold', '0.0'))
            age_bucket_method = props.get('age_bucket_method')
            age_buckets = int(props.get('age_buckets','3')) if props.get('age_buckets') else 3
            try:
                metric_result = evaluate_group_min_positive_rate(df, target_col, dimension, thresh, age_bucket_method, age_buckets)
            except Exception as e:
                metric_result = {"metric_key": metric, "actual_value": None, "threshold": thresh, "operator": "gt", "passed": False, "metadata": {"error": str(e)}}
            # decorate with control info
            res = {
                "control_id": r.get('control-id') or r.get('control_id') or r.get('control-id'),
                "description": r.get('description',''),
                **metric_result,
                "severity": props.get('severity','medium')
            }
            results.append(res)
    return results
