r"""Declarative registered-report dispatch seam (audit finding PB-07).

``vfe3.viz.figures`` has a ``register_figure``/``get_figure`` registry that production report
drivers never consulted: a driver could only call a figure generator it imported by name, so the
registry was write-only. ``FigureSpec`` declares one figure by its registry name, its output
filename, and an adapter that maps a shared context mapping to that generator's kwargs (or
``None`` to skip); ``emit_registered_figures`` dispatches a list of specs through
``vfe3.viz.figures.get_figure`` with per-spec failure isolation, atomic tmp-file publication (a
build failure or crash cannot corrupt an existing output), and figure-leak cleanup.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence
from uuid import uuid4

import matplotlib.pyplot as plt

from vfe3.viz.figures import get_figure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FigureSpec:
    name:        str
    output_name: str
    adapter:     Callable[[Mapping[str, object]], Optional[Mapping[str, object]]]


def emit_registered_figures(
    specs:      Sequence[FigureSpec],
    context:    Mapping[str, object],
    output_dir: Path,
) -> List[Path]:
    names = [spec.output_name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("registered figure output names must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in specs:
        before = set(plt.get_fignums())
        fig = None
        tmp_target = None
        try:
            kwargs = spec.adapter(context)
            if kwargs is None:
                continue
            target = output_dir / spec.output_name
            tmp_target = output_dir / (
                f".{target.stem}.{uuid4().hex}.tmp{target.suffix}"
            )
            fig = get_figure(spec.name)(**dict(kwargs), path=str(tmp_target))
            if not tmp_target.is_file() or tmp_target.stat().st_size == 0:
                raise RuntimeError(
                    f"figure {spec.name!r} did not write its temporary output"
                )
            os.replace(tmp_target, target)
            written.append(target)
        except Exception as exc:
            logger.warning("registered figure %s skipped: %s", spec.name, exc)
        finally:
            if tmp_target is not None:
                tmp_target.unlink(missing_ok=True)
            if fig is not None:
                plt.close(fig)
            for number in set(plt.get_fignums()) - before:
                plt.close(number)
    return written
