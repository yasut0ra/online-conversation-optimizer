# Online Conversation Optimizer


MVP that generates N reply candidates, annotates features, exposes action propensities, and learns online with a contextual bandit. Safe-by-default, evaluation-ready (IPS/DR logs).


## Run (pseudo)
export OPENAI_API_KEY=...
python src/app.py


## Logs
Each turn logs: {context_hash, candidates, chosen_idx, propensity, reward, features}.