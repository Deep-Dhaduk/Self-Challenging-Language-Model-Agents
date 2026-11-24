# Self-Challenging Language Model Agents

An individual short story assignment exploring **[Self-Challenging Language Model Agents](https://arxiv.org/abs/2506.01716)** by Yifei Zhou, Sergey Levine, Jason Weston, Xian Li, and Sainbayar Sukhbaatar (2025).

The project combines an original research article, a PowerPoint presentation, an executed Google Colab notebook, and an educational implementation of tool-using agents and task verification.

## Project links

| Material | Link |
| --- | --- |
| Research paper | [Self-Challenging Language Model Agents](https://arxiv.org/abs/2506.01716) |
| Published Medium article | [Can AI Agents Learn by Creating Their Own Challenges?](https://medium.com/@deepbharatbhai.dhaduk/can-ai-agents-learn-by-creating-their-own-challenges-dd7cdbf376e7) |
| Final executed notebook | [View notebook with saved outputs](notebooks/self_challenging_agents.ipynb) |
| Google Colab | [Open the final notebook in Colab](https://colab.research.google.com/github/Deep-Dhaduk/Self-Challenging-Language-Model-Agents/blob/main/notebooks/self_challenging_agents.ipynb) |
| Presentation | [Download PowerPoint](slides/self_challenging_agents.pptx) |
| Repository | [GitHub project](https://github.com/Deep-Dhaduk/Self-Challenging-Language-Model-Agents) |

The presentation contains 18 slides, editable diagrams and charts, embedded speaker notes, and source links.

## Research focus

The selected paper investigates agents that create practice tasks and learn from execution feedback. The companion implementation explores:

- A challenger that explores a simulated tool environment.
- Tasks containing an instruction, verifier checks, a reference solution, and failure cases.
- Filtering that checks reference solutions and rejects known failure cases.
- Executors that interact with tools using observations from earlier actions.
- Evaluation of task outcomes, action errors, and verifier reliability.

Our simulated retail environment supports customer lookup, order inspection, cancellation, returns, and address changes. Verifier specifications are interpreted by trusted Python code. The implementation is an educational adaptation; its results are separate from the paper's M3ToolEval and TauBench results.

## Final Colab notebook

[self_challenging_agents.ipynb](notebooks/self_challenging_agents.ipynb) is the original executed notebook, preserved with its code, cell outputs, charts, and experiment summary. It is the only file in `notebooks/`.

The notebook records the CPU demonstration and a Qwen2.5-0.5B-Instruct tool-use experiment with template-generated tasks. The saved Colab run used [source revision 9d56c24](https://github.com/Deep-Dhaduk/Self-Challenging-Language-Model-Agents/tree/9d56c245600660979e69c014acc5c1e9d07ba376). Its saved outputs document that run; the source package in the repository also contains later interface refinements.

## Recorded results

### CPU demonstration

The CPU experiment updates a categorical policy over five hand-written tool-use strategies. It uses 120 training tasks, four attempts per training task, and 60 held-out test tasks with four attempts each, using seed 42.

| Metric | Before update | After update |
| --- | ---: | ---: |
| Pass@1 | 23.3% | 86.7% |
| Pass@4 | 73.3% | 100.0% |
| Tool/action errors across 240 evaluation attempts | 75 | 7 |

Of 480 training attempts, 104 received a reward. These results measure selection among existing scripted strategies; they are not LLM fine-tuning results. See the [configuration, metrics, and example traces](results/cpu_demo/README.md).

### LLM execution

The final notebook's 0.5B experiment evaluated 12 held-out tasks and collected 96 training attempts. None received a reward, so rejection fine-tuning was skipped and there is no after-training score. The logged failures include invalid tool calls, invented order IDs, and missed lookup steps.

The [original results ZIP](results/llm_first_attempt/original_run.zip), [run summary](results/llm_first_attempt/summary.json), [configuration](results/llm_first_attempt/config.json), and [diagnostics](results/llm_first_attempt/diagnostics.json) preserve the evidence.

An additional 1.5B diagnostic recorded 0 successful tasks out of three readiness probes, before training or held-out evaluation. Its [summary](results/llm_second_attempt/summary.json), [recovered trace](results/llm_second_attempt/first_preflight_trace.txt), and [provenance](results/llm_second_attempt/provenance.json) are retained as supporting experiment records. This diagnostic is distinct from the final notebook's original 0.5B run.

### Task-filter ablation

A controlled dataset contains 48 fixtures: 12 valid tasks and 12 each with bad reference solutions, permissive verifiers, or overstrict verifiers.

| Filtering level | Fixtures accepted |
| --- | ---: |
| Schema only | 48/48 |
| Reference-solution check | 24/48 |
| Reference, no-op, and failure-case checks | 12/48 |

These are deliberately injected defects. The [ablation results](results/cpu_demo/filter_ablation.json) measure filter behavior on this fixture set, rather than an LLM's task-generation error rate.

## Run the CPU demonstration

Python 3.10+ is required. This path uses only the standard library, with no model download, GPU, or API key.

```bash
git clone https://github.com/Deep-Dhaduk/Self-Challenging-Language-Model-Agents.git
cd Self-Challenging-Language-Model-Agents
python -m sca demo --output outputs/demo
```

The command saves task specifications, conversations, metrics, policy weights, and charts under `outputs/demo/`. Use a new output directory for each run. Generated runs, caches, and model weights are excluded from Git.

The package also includes Hugging Face inference, native function-call formatting, an optional LLM challenger, and response-only LoRA training utilities. Their dependencies are defined in the `llm` optional dependency group in [pyproject.toml](pyproject.toml). The recorded results above identify what the submitted experiments measured.

## Evaluation notes

- **Pass@1:** fraction of test tasks solved on the first sampled attempt.
- **Pass@k:** fraction solved at least once among k sampled attempts; this does not mean every attempt succeeds.
- **Tool/action errors:** unsuccessful tool calls and malformed action responses.
- **Verifier/oracle disagreement:** a difference between the task verifier's reward and the independent goal evaluator.

The evaluator checks the requested change and preservation of unrelated orders. Training and test tasks are kept separate. The executor receives the request, tool information, and observations, without the task's reference solution, verifier, or failure cases.

The synthetic task families and small sample sizes limit generalization. One seed does not establish statistical significance, and finite failure-case tests do not prove a verifier correct.

## Tests

Core checks run without optional packages:

```bash
python -m unittest discover -s tests -p test_core.py -v
```

With the optional LLM dependencies installed:

```bash
python -m unittest discover -s tests -v
```

The checks cover environment state, filtering, train/test separation, preservation of the original notebook outputs, function-call formatting, response-only masking, and tiny-model LoRA updates and adapter reloading. Tiny random-model checks validate code behavior; they are separate from pretrained-model performance measurements.

## Repository structure

```text
README.md
pyproject.toml
notebooks/
  self_challenging_agents.ipynb   # Final executed Colab notebook
slides/
  self_challenging_agents.pptx   # PowerPoint with embedded speaker notes
sca/                            # Environment, agents, verification, training, and reporting
tests/                          # Core and optional model checks
results/
  cpu_demo/                     # Measured policy-learning demonstration
  llm_first_attempt/            # Final notebook's original experiment artifacts
  llm_second_attempt/           # Supporting diagnostic evidence
```

## Reference

```bibtex
@misc{zhou2025selfchallenging,
  title={Self-Challenging Language Model Agents},
  author={Yifei Zhou and Sergey Levine and Jason Weston and Xian Li and Sainbayar Sukhbaatar},
  year={2025},
  eprint={2506.01716},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.01716}
}
```
