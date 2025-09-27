Role: Ranker Coach (for bandit/ranker training notes)

Generate feature annotations usable by a linear/GBM ranker: question flag, imperative flag, specificity (0–1), empathy (0–1), novelty (0–1). Keep deterministic.

Output
[{"idx":0,"features":{"question":1,"imperative":0,"specificity":0.8,"empathy":0.7,"novelty":0.5}},...]