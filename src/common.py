from typing import Dict


def assign_label(label: str, classification_pattern: int) -> str:
    """
    Returns a label for each classification task.
    """
    if classification_pattern == 1:
        if label == 'EBV':
            return 'EBV'
        elif label == 'MSI':
            return 'MSI'
        else:
            return 'other'
    elif classification_pattern == 2:
        if label in {'EBV', 'MSI'}:
            return 'EBV_MSI'
        else:
            return 'other'
    elif classification_pattern == 3:
        if label == 'EBV':
            return 'EBV'
        else:
            return 'MSI_other'
    elif classification_pattern == 4:
        if label == 'MSI':
            return 'MSI'
        else:
            return 'EBV_other'


def get_dict_labels(classification_pattern: int) -> Dict[str, int]:
    """
    Returns a dictionary that assigns a number to molecular classification.
    """
    if classification_pattern == 1:
        return {'EBV': 0, 'MSI': 1, 'other': 2}
    elif classification_pattern == 2:
        return {'EBV_MSI': 0, 'other': 1}
    elif classification_pattern == 3:
        return {'EBV': 0, 'MSI_other': 1}
    elif classification_pattern == 4:
        return {'MSI': 0, 'EBV_other': 1}
