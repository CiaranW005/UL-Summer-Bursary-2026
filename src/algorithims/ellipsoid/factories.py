from .algorithim import EllipsoidCover
from .evaluator import EllipsoidEvaluator

from .fitter import EllipsoidFitter
from .cleaner import CandidateCleaner
def create_cover() -> EllipsoidCover:
    fitter = EllipsoidFitter(
        support_points=5,
        reg=1e-4,
    )

    cleaner = CandidateCleaner(
        min_points=1,
        fitter=fitter,
    )

    return EllipsoidCover(
        fitter=fitter,
        cleaner=cleaner,
    )


def create_evaluator() -> EllipsoidEvaluator:
    return EllipsoidEvaluator()