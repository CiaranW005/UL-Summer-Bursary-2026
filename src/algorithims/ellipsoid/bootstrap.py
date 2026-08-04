
import numpy as np
import pandas as pd 

import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm

from collections.abc import Callable

from .algorithim import EllipsoidCover
from .evaluator import EllipsoidEvaluator

CoverFactory = Callable[[], EllipsoidCover]
EvalFactory = Callable[[], EllipsoidEvaluator]

_WORKER_TRAIN: np.ndarray | None = None
_WORKER_GOOD: np.ndarray | None = None
_WORKER_DEFECT: np.ndarray | None = None

_WORKER_SHM: list[shared_memory.SharedMemory] = []

def create_shared_array(
    array: np.ndarray
) -> tuple[shared_memory.SharedMemory, tuple[int, ...], str]:
    contiguous = np.ascontiguousarray(array)

    shm = shared_memory.SharedMemory(
        create=True,
        size=contiguous.nbytes
    )

    shared_view = np.ndarray(
        contiguous.shape,
        dtype=contiguous.dtype,
        buffer=shm.buf
    )

    shared_view[:] = contiguous

    return shm, contiguous.shape, contiguous.dtype.str

def _init_worker(
    train_name: str,
    train_shape: tuple[int, ...],
    train_dtype: str,
    good_name: str,
    good_shape: tuple[int, ...],
    good_dtype: str,
    defect_name: str,
    defect_shape: tuple[int, ...],
    defect_dtype: str
) -> None:
    global _WORKER_TRAIN
    global _WORKER_GOOD
    global _WORKER_DEFECT
    global _WORKER_SHM

    train_shm = shared_memory.SharedMemory(name=train_name)
    good_shm = shared_memory.SharedMemory(name=good_name)
    defect_shm = shared_memory.SharedMemory(name=defect_name)

    _WORKER_SHM = [
        train_shm,
        good_shm,
        defect_shm
    ]

    _WORKER_TRAIN = np.ndarray(
        train_shape,
        dtype=np.dtype(train_dtype),
        buffer=train_shm.buf
    )

    _WORKER_GOOD = np.ndarray(
        good_shape,
        dtype=np.dtype(good_dtype),
        buffer=good_shm.buf
    )

    _WORKER_DEFECT = np.ndarray(
        defect_shape,
        dtype=np.dtype(defect_dtype),
        buffer=defect_shm.buf
    )

    _WORKER_TRAIN.flags.writeable = False
    _WORKER_GOOD.flags.writeable = False
    _WORKER_DEFECT.flags.writeable = False

def _run_train_bootstrap(
    bootstrap_indices: list[int],
    seeds: list[int],
    cover_factory: CoverFactory,
    eval_factory: EvalFactory,
)-> list[dict[str, float | int]]:
    if (
        _WORKER_TRAIN is None 
        or _WORKER_GOOD is None
        or _WORKER_DEFECT is None
    ):
        raise RuntimeError(
            "Worker embeddings were not initialised before creating the pool"
        )
    
    cover = cover_factory()
    evaluator = eval_factory()

    results: list[dict[str, float | int]] = []

    for idx, seed in tqdm(zip(
            bootstrap_indices,
            seeds,
            strict=True
        ),
        desc=f"Train Bootstrap [{bootstrap_indices[0]}-{bootstrap_indices[-1]}]",
        position=bootstrap_indices[0] // max(1, len(bootstrap_indices)),
        total=len(bootstrap_indices),
        leave=False
    ):
        rng = np.random.default_rng(seed)

        sample_idx = rng.choice(
            len(_WORKER_TRAIN),
            len(_WORKER_TRAIN),
            replace=True
        )
  
        train_bootstrap = _WORKER_TRAIN[sample_idx]
        ellipsoids, _ = cover.run(embeds=train_bootstrap)

        _, metrics = evaluator.evaluate_detection(
            good_test_emb=_WORKER_GOOD,
            defect_test_emb=_WORKER_DEFECT,
            ellipsoids=ellipsoids
        )
        
        results.append({
            "bootstrap": idx,
            "auroc": float(metrics["auroc"]),
            "n_ellipsoids": len(ellipsoids),
        })

    return results

class BootstrapRunner:
    def __init__(self, 
                cover_factory: CoverFactory, 
                eval_factory: EvalFactory, 
                n_test_bootstraps: int = 1000,
                n_train_bootstraps: int = 100, 
                seed: int = 42,
                max_workers: int = 2
        ) -> None:
        self.cover_factory = cover_factory
        self.eval_factory = eval_factory

        self.test_bootstraps = n_test_bootstraps
        self.train_bootstraps = n_train_bootstraps

        self.seed = seed
        self.max_workers = max_workers

    def run(self,
            train_emb: np.ndarray,
            good_test_emb: np.ndarray,
            defect_test_emb: np.ndarray
        )-> tuple[pd.DataFrame, pd.DataFrame]:
        test_df = self.bootstrap_test_embeds(
            train_emb=train_emb,
            good_test_emb=good_test_emb,
            defect_test_emb=defect_test_emb
        )

        train_df = self.bootstrap_train_embeds(
            train_emb=train_emb,
            good_test_emb=good_test_emb,
            defect_test_emb=defect_test_emb
        )

        return test_df, train_df

    def bootstrap_test_embeds(self, 
            train_emb: np.ndarray, 
            good_test_emb: np.ndarray,
            defect_test_emb: np.ndarray
        )-> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        results: list[float] = []

        cover = self.cover_factory()
        eval = self.eval_factory()

        ellipsoids, _ = cover.run(embeds=train_emb)
        for _ in tqdm(range(self.test_bootstraps), desc="Test Bootstraps"):
            # Good and test ratios should remain the same
            good_idx = rng.choice(
                len(good_test_emb),
                len(good_test_emb),
                replace=True
            )

            defect_idx = rng.choice(
                len(defect_test_emb),
                len(defect_test_emb),
                replace=True
            )

            good_boot_emb = good_test_emb[good_idx]
            defect_boot_emb = defect_test_emb[defect_idx]

            _, metrics = eval.evaluate_detection(
                good_test_emb=good_boot_emb,
                defect_test_emb=defect_boot_emb,
                ellipsoids=ellipsoids
            )

            results.append(metrics["auroc"])

        return pd.DataFrame({
            "bootstrap": np.arange(self.test_bootstraps),
            "auroc": results,
        })

    def bootstrap_train_embeds(self,
            train_emb: np.ndarray,
            good_test_emb: np.ndarray,
            defect_test_emb: np.ndarray
        )-> pd.DataFrame:
        seed_seq = np.random.SeedSequence(self.seed)

        seeds = [
            int(child.generate_state(1)[0])
            for child in seed_seq.spawn(self.train_bootstraps)
        ]

        bootstrap_indices = np.arange(self.train_bootstraps)

        index_chunks = [
        chunk.tolist()
        for chunk in np.array_split(
            bootstrap_indices,
            self.max_workers
        )
        if len(chunk) > 0
        ]

        seed_chunks = [
            chunk.tolist()
            for chunk in np.array_split(
                seeds,
                self.max_workers
            )
            if len(chunk) > 0
        ]

        train_shm, train_shape, train_dtype = create_shared_array(train_emb)
        good_shm, good_shape, good_dtype = create_shared_array(good_test_emb)
        defect_shm, defect_shape, defect_dtype = create_shared_array(defect_test_emb)

        results: list[dict[str, float | int]] = []
        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context("forkserver"),
                initializer=_init_worker,
                initargs=(
                    train_shm.name,
                    train_shape,
                    train_dtype,
                    good_shm.name,
                    good_shape,
                    good_dtype,
                    defect_shm.name,
                    defect_shape,
                    defect_dtype,
                )
            ) as executor:
                futures = [
                    executor.submit(
                        _run_train_bootstrap,
                        index_chunk,
                        seed_chunk,
                        self.cover_factory,
                        self.eval_factory,
                    )
                    for index_chunk, seed_chunk in zip(
                        index_chunks,
                        seed_chunks,
                        strict=True,
                    )
                ]

                for future in as_completed(futures):
                    results.extend(future.result())
        finally:
            train_shm.close()
            train_shm.unlink()

            good_shm.close()
            good_shm.unlink()

            defect_shm.close()
            defect_shm.unlink()

        return (
            pd.DataFrame(results)
            .sort_values("bootstrap")
            .reset_index(drop=True)
        )

    @staticmethod
    def summarise_bootstrap(df: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "mean": df["auroc"].mean(),
            "std": df["auroc"].std(),
            "ci_lower": df["auroc"].quantile(0.025),
            "ci_upper": df["auroc"].quantile(0.975),
        })


