install:; pip install -r requirements.txt && pip install -e .
test:; pytest -q
demo:; python scripts/demo.py --model mobilenet --budget 50
quick:; python -m experiments.run_experiments --models mobilenet pipeline --budgets 10 25 50 --seeds 3 --out results
full:; python -m experiments.run_experiments --budgets 10 25 50 100 --seeds 5 --out results
