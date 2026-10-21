"""Pure geometry for conservative right-side text capacity."""
from statistics import median

GAP_PT = 2.0
ALIGN_PT = 3.0
NEIGHBOR_DISTANCE_PT = 120.0


def column_right(rect: list[float], boxes: list[list[float]], page_width: float) -> float:
    x0, y0, x1, y1 = rect
    # Existing text widths are preferences, not barriers to verified blank space.
    right = page_width - GAP_PT
    local = [r for r in boxes if r[3] >= y0 - NEIGHBOR_DISTANCE_PT and r[1] <= y1 + NEIGHBOR_DISTANCE_PT]
    starts = sorted(r[0] for r in local if r[0] > x0 + ALIGN_PT)
    clusters: list[list[float]] = []
    for start in starts:
        if clusters and start - clusters[-1][-1] <= ALIGN_PT:
            clusters[-1].append(start)
        else:
            clusters.append([start])
    next_columns = [median(group) - GAP_PT for group in clusters if len(group) >= 3 and median(group) >= x1]
    if next_columns:
        right = min(right, min(next_columns))
    for r in boxes:
        if min(y1, r[3]) <= max(y0, r[1]):
            continue
        if r[0] >= x1:
            right = min(right, r[0] - GAP_PT)
        elif r[2] > x1 and abs(r[0] - x0) > ALIGN_PT:
            return x1
    return max(x1, right)
