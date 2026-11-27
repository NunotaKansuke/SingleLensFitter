"""Replay singlens compare-single-vbml magnification benchmarks, one engine per process."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def timer(call):
    start = perf_counter()
    result = np.asarray(call())
    first = perf_counter() - start
    if not np.isfinite(result).all():
        raise ValueError("Non-finite light curve")
    for _ in range(3):
        call()
    samples = []
    for _ in range(100):
        start = perf_counter()
        call()
        samples.append(perf_counter() - start)
    return dict(first_seconds=first, seconds=samples, median_us=np.median(samples)*1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["native", "table", "fft", "vbm"])
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "native":
        from singlelensfitter import fitters
        assert fitters._vbm_cpp is not None
    elif args.engine == "vbm":
        import VBMicrolensing
    else:
        import jax
        import jax.numpy as jnp
        import singlens as sl
        jax.config.update("jax_enable_x64", True)
    rows = []
    for source, a in [("uniform", 0.), ("linear", .5)]:
        for n in (10, 30, 100, 300, 1000):
            t = np.linspace(-1., 1., n)
            if args.engine == "native":
                model = fitters.FSPLFitter(linear_limb_darkening=a)
                q = np.array([0., 20., .005, np.log(.01)])
                def call():
                    return model._magnification(t, q)
            elif args.engine == "vbm":
                model = VBMicrolensing.VBMicrolensing()
                model.Tol = model.RelTol = 1e-3
                model.a1 = a
                model.minannuli = 2 if a else 1
                theta = [np.log(.005), np.log(20.), 0., np.log(.01)]
                times = t.tolist()
                batch = model.ESPLLightCurve
                def call():
                    return batch(theta, times)[0]
            else:
                model = sl.Model(source=source, **({"solver": "fft"} if args.engine == "fft" else {}))
                compiled = jax.jit(model.magnification)
                p = sl.Parameters(t0=0., t_e=20., u0=.005, rho=.01, a1=a if a else None)
                times = jnp.asarray(t)
                jax.block_until_ready((times, p))
                def call():
                    return compiled(times, p).block_until_ready()
            rows.append(dict(source=source, n=n, **timer(call)))
    (args.output / f"{args.engine}.json").write_text(json.dumps(rows, indent=2))
    if args.engine != "native":
        return
    reference = args.upstream / "example/compare-single-vbml/outputs/accuracy.csv"
    with reference.open() as f:
        points = [r for r in csv.DictReader(f) if r["implementation"] == "singlens table"]
    with (args.output / "accuracy.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["source","rho","u_over_rho","reference","native","relative_error"])
        for source, rho in dict.fromkeys((r["source"], r["rho"]) for r in points):
            group = [r for r in points if (r["source"], r["rho"]) == (source, rho)]
            z = np.array([float(r["u_over_rho"]) for r in group])
            expected = np.array([float(r["reference"]) for r in group])
            actual = fitters._cpp_fspl_magnification(z*float(rho), float(rho),
                espl_table_path=model.espl_table_path, tol=model.magnification_tol,
                reltol=model.magnification_reltol, linear_limb_darkening=.5 if source=="linear" else 0.)
            writer.writerows(zip([source]*len(z), [rho]*len(z), z, expected, actual, abs(actual/expected-1)))


if __name__ == "__main__":
    main()
