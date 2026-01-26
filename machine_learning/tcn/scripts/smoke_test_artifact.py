import argparse
from machine_learning.artifacts import smoke_test_artifact_determinism

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--fill_value", type=float, default=0.0)
    args = ap.parse_args()

    res = smoke_test_artifact_determinism(
        args.run_dir,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        n_samples=args.n,
        atol=args.atol,
        rtol=args.rtol,
        strict=args.strict,
        fill_value=args.fill_value,
    )
    print("SMOKE TEST OK:", res)

if __name__ == "__main__":
    main()
