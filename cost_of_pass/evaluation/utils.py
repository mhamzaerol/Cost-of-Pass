from collections import defaultdict
from typing import List, Tuple, Optional
from cost_of_pass.evaluation.recording import BaseRecord

def group_records_by_input_idx(records: List[BaseRecord]) -> Tuple[List[BaseRecord], List[BaseRecord], bool]:
    """
    Groups the records by input_idx and separates them into good and bad records.
    """
    good_per_idx = defaultdict(list)
    bad_per_idx = defaultdict(list)
    has_bad = False
    
    for record in records:
        if record.completed:
            good_per_idx[record.input_idx].append(record)
        else:
            bad_per_idx[record.input_idx].append(record)
            has_bad = True
    
    return good_per_idx, bad_per_idx, has_bad

def group_and_trim_records(records: List[BaseRecord], n_runs: int) -> Tuple[List[BaseRecord], List[BaseRecord]]:
    """
    Groups the records by input_idx and trims the records to n_runs.
    """
    good_per_idx, bad_per_idx, _ = group_records_by_input_idx(records)
    
    ret_records, rem_records = [], []
    for input_idx in set(good_per_idx.keys()) | set(bad_per_idx.keys()):
        combined_records = good_per_idx.get(input_idx, []) + \
                            bad_per_idx.get(input_idx, [])
        ret_records += combined_records[:n_runs]
        rem_records += combined_records[n_runs:]

    return ret_records, rem_records

def expected_value(x: List[float], p: Optional[List[float]] = None) -> float:
    """
    Computes the expected value of a list of values.
    """
    if p is None:
        return sum(x) / len(x)
    else:
        return sum([x[i] * p[i] for i in range(len(x))]) / sum(p)