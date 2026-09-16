import sys
from evaluate_baselines import main


if __name__ == "__main__":
    if "--years" not in sys.argv:
        sys.argv += ["--years", "2016", "2017", "2018", "2019"]
    if "--output" not in sys.argv:
        sys.argv += ["--output", "metrics/baselines_2016_2019.json"]
    main()
