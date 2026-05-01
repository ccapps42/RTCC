"""Curriculum phase scheduler — controls seq_len, loop count, and dataset per phase."""
from dataclasses import dataclass


@dataclass
class Phase:
    phase_num: int
    start_step: int
    end_step: int
    seq_len: int
    loop_min: int
    loop_max: int
    dataset: str  # key into loader.DATASETS


# Curriculum from plan — same as OpenHobbs, validated
CURRICULUM = [
    Phase(1, 0,      7_500,  128, 2, 4, "tinystories"),
    Phase(2, 7_500,  22_500, 256, 2, 6, "wikipedia"),
    Phase(3, 22_500, 52_500, 512, 2, 8, "fineweb_edu"),
    Phase(4, 52_500, 75_000, 512, 2, 8, "fineweb"),
]


def get_phase(step: int) -> Phase:
    for phase in CURRICULUM:
        if phase.start_step <= step < phase.end_step:
            return phase
    return CURRICULUM[-1]  # stay in phase 4 if beyond total_steps


def get_loop_count(step: int, rng=None) -> int:
    """Sample loop count uniformly in [loop_min, loop_max] for current phase."""
    import random
    phase = get_phase(step)
    if rng is None:
        return random.randint(phase.loop_min, phase.loop_max)
    return rng.randint(phase.loop_min, phase.loop_max)
