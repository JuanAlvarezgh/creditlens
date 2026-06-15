def compute_total_late_payments(times_30_59: int, times_60_89: int, times_90: int) -> int:
    return times_30_59 + times_60_89 + times_90


def compute_utilization_segment(utilization: float) -> str:
    if utilization <= 0.3:
        return "low"
    elif utilization <= 0.7:
        return "medium"
    return "high"


def compute_dti(debt_ratio: float, monthly_income: float) -> float:
    if monthly_income <= 0:
        return 0.0
    return debt_ratio


def test_total_late_payments_sums_all_buckets():
    assert compute_total_late_payments(2, 1, 0) == 3


def test_total_late_payments_all_zeros():
    assert compute_total_late_payments(0, 0, 0) == 0


def test_utilization_segment_low():
    assert compute_utilization_segment(0.2) == "low"


def test_utilization_segment_boundary_low():
    assert compute_utilization_segment(0.3) == "low"


def test_utilization_segment_medium():
    assert compute_utilization_segment(0.5) == "medium"


def test_utilization_segment_high():
    assert compute_utilization_segment(0.9) == "high"


def test_dti_normal():
    assert compute_dti(0.3, 5000) == 0.3


def test_dti_zero_income_returns_zero():
    assert compute_dti(0.5, 0) == 0.0
