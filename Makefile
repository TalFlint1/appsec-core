.PHONY: corpus-stats eval-hit-rate eval-hallu eval-memory

corpus-stats:
\tpython scripts/corpus_stats.py

# Use: make eval-hit-rate K=5 ABLATION=baseline|current
eval-hit-rate:
\tpython scripts/eval_hit_rate.py --k $(if $(K),$(K),5) --ablation $(if $(ABLATION),$(ABLATION),current)

# Run twice: gating=off and gating=on
eval-hallu:
\tpython scripts/eval_hallucination.py --gating on
\tpython scripts/eval_hallucination.py --gating off

eval-memory:
\tpython scripts/eval_memory_ab.py
