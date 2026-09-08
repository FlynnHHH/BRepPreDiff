# Fusion360Seg 97% accuracy iteration log

Started 2026-09-07. User-authorized protocol: Fusion360Seg s2.0.0 only,
50 pretraining epochs, 100 fine-tuning epochs, GPUs 0–4. These explicit epoch
budgets override the project-wide default of 200 downstream epochs.

## Evaluation protocol

- Existing clean splits: 24,964 training CADs, 5,350 validation CADs, 5,366 test CADs.
- Split identifiers are unique and mutually disjoint. Train/validation are subsets
  of official training identifiers; test is a subset of official test identifiers.
- Pretraining uses training geometry only, with labels stripped and val/test disabled.
- Fine-tuning uses training labels. Best checkpoint is selected by validation face accuracy.
- Search uses validation metrics. The independent test set is reserved for a
  validation-selected candidate. No test-set class bias fitting or test-driven selection.
- Target means face-level accuracy, matching the existing project metric. A
  validation result alone does not count as achieving the 97% test target.
- Seed 42 screening; promising methods should receive additional seed checks.
- Per-run source snapshots, hashes, exact command lines, embedded checkpoint configs,
  logs and validation evaluations are saved in `runs/fusion97`.

## Round 1: independent encoder hypotheses

All runs use full pretraining objective, LR 1e-4 constant, batch 128; MLP fine-tuning
uses LR 3e-4, batch 256, CE + 0.3 Dice, and per-graph preprocessing.

| GPU | Variant | Change | Status |
|---|---|---|---|
| 0 | baseline | 128 dimensions, 4 edge-update attention layers | Running |
| 1 | masked | Mask 50% of categorical attributes; reverse/parallel edges masked together | Running |
| 2 | wide | 256 hidden dimensions | Running |
| 3 | grid | Additional 2D face-grid and 1D edge-grid encoders | Running |
| 4 | context | Learned combination of layer features plus graph mean/max context | Running |

The grid branch treats trim mask as an input channel; it does not apply a hard
mask to the noisy pretraining input. The masked branch removes categorical
embeddings and computes categorical losses only at masked positions.

## Prepared next hypotheses

- `geometric`: isotropic per-CAD coordinate normalization; unit normals/tangents;
  dimensionless area, length and curvature; exact trim masks. Intended to preserve
  relationships destroyed by separate normalization of flattened UV slots.
- `geometric_grid`: combine the physical coordinate frame with local grid encoding.
- `boundary`: auxiliary prediction of label changes across adjacent faces.
- `boundary_refine`: additionally gate a residual neighbor aggregation using predicted boundaries.
- `deep`: eight attention layers as a capacity/receptive-field control.

These are hypotheses, not claimed accuracy improvements. Round 1 source snapshots
are isolated from subsequent development.

## Verification so far

- Initial encoder/masking and label-diffusion regression subset: 16 passed.
- Additional physical-normalization invariance, boundary gradients, preprocessing,
  cached-dataset and segmentation regression subset: 24 passed.
- CUDA access requires execution outside the read-only GPU sandbox; GPUs 0–4 were
  idle at launch. Existing jobs on GPUs 5–7 were not modified.

## Progress at approximately 21:35

All five pretraining runs completed exactly 50 epochs. Fine-tuning is still in
progress (approximately epochs 42–46/100). Best validation accuracies so far:
baseline 96.020%, masked 95.931%, wide 96.039%, grid 95.868%, context 96.130%.
These are provisional, different-epoch maxima, not final comparisons.

Round 2 queues wait for their corresponding GPU's first experiment to finish:

| GPU | Next variant | Encoder |
|---|---|---|
| 0 | geometric | New 50-epoch pretraining |
| 1 | boundary | Reuse round-1 baseline's 50-epoch encoder |
| 2 | localedge | New 50-epoch pretraining on local-edge geometry |
| 3 | geometric_grid | New 50-epoch pretraining |
| 4 | boundary_refine | Reuse round-1 baseline's 50-epoch encoder |

The unstarted deep queue was replaced with localedge; no running training was interrupted.

Further implemented but not launched: joint graph-label diffusion (complete-label
graphs, one noise realization, shared per-graph timestep, five sampling steps),
finetuning-only rigid rotations, finetuning-only UV grid symmetries, and optional
cosine learning-rate decay. Train/validation divergence motivates the augmentation
candidates; they are not yet demonstrated improvements.

### Original B-Rep geometry

Raw source: `/home/nvme03/hhfeng/fusion360segmentationdataset/s2.0.0/`.
Separate cache: `cache/features/fusion360seg_localedge_v1`.
Local pcurve normals replace only angle/dot/relation features. Faces, labels,
edge ordering, lengths and edge sample grids are preserved. The computation is
unsigned and does not yet distinguish convex from concave edges.

100-CAD pilot: 3,476 successful edge pairs, 11 fallbacks, zero errors.
Full train/val build: 30,314 CADs, zero errors; 990,196 successful pairs and 14,917
fallbacks among newly processed CADs (the pilot's 100 CADs were reused).
Face counts, raw/cache labels and edge ordering were checked. Original caches were
not overwritten. No local-edge test caches have been built yet.

Complete regression suite after these additions: **139 passed**.
The test accuracy target has not been achieved or evaluated yet.

## Completed rounds and GPU restriction (2026-09-08)

The user now restricts all further experiments to **GPU 4 only**. This supersedes
the original GPU 0–4 allocation. Both earlier rounds completed; no campaign
training or queue processes were live when checked. Unrelated GPU 0–3 jobs were
left untouched. The launcher now accepts only GPU 4.

Final validation accuracies (all 100 fine-tuning epochs completed):

| Variant | Validation accuracy (%) |
|---|---:|
| baseline | 96.197 |
| masked | 96.079 |
| wide | 96.259 |
| grid | 96.086 |
| context | 96.389 |
| boundary | 96.211 |
| boundary_refine | 96.132 |
| geometric | 95.662 |
| geometric_grid | 95.973 |
| localedge | 96.170 |

The context variant is the strongest current candidate. Its independently
rerun validation evaluation reports exactly 0.9638888888888889 over 79,920 faces
from 5,350 CADs, using its validation-selected epoch-97 checkpoint.
Geometry normalization and local-edge corrections did not improve the baseline
under the screened protocol; boundary supervision gave only a small single-seed
gain. Next, run context + cosine fine-tuning and context + rotation fine-tuning
sequentially on GPU 4, reusing the same completed 50-epoch context encoder.
Test labels/predictions have not been used for candidate selection.

### Validation-error-driven operation supervision

Inspection of the completed context model's validation confusion matrix found
693 CutSide→ExtrudeSide and 419 ExtrudeSide→CutSide mistakes: 1,112 of 2,886
errors (38.53%). This motivates a new `context_operation` auxiliary head that
groups the eight labels into Extrude, Cut, Fillet, Chamfer and Revolve families.
The mapping is explicitly configured as `[0,0,1,1,2,3,4,4]`; auxiliary CE has weight
0.2, uses training labels only, and excludes ignored faces. Inference still uses
the original eight-class head. It is an unproven hypothesis rather than a claimed gain.

The cosine training PID 2600501 and rotation queue PID 2601288 were verified live
on 2026-09-08; no restart was performed. The operation candidate is queued after
rotation on GPU 4, using the same completed context pretraining checkpoint and
100 fine-tuning epochs. Operation-head gradients, ignored-label handling and
ordinary inference were tested; the complete suite passed **140 tests**.

## Round 3 final results and frozen test evaluation

All three round-3 candidates completed 100 fine-tuning epochs on GPU 4:

| Variant | Best epoch | Validation accuracy (%) |
|---|---:|---:|
| context_cosine | 63 | 96.240 |
| context_operation | 82 | 96.299 |
| context_rotation | 92 | **96.906** |

`context_rotation` was selected using validation only. After freezing that choice,
its epoch-92 checkpoint was evaluated once on the official test split: 5,366 CADs,
77,070 faces, **96.593% face accuracy**. This is below the 97% target, so no further
test-guided candidate selection is performed. The next validation-only experiment
combines rotation augmentation with operation-family auxiliary supervision and
reuses the exact completed 50-epoch context pretraining checkpoint.

A second queued candidate, `context_rotation_mix`, samples a rigid rotation for
50% of training CADs and retains the canonical frame for the other 50%. This tests
whether the always-rotated policy over-shifts training away from the canonical
validation/test distribution while retaining its demonstrated regularization gain.
It reuses the same 50-epoch context encoder and runs only after the operation
candidate finishes on GPU 4.

While that GPU-4 run is active, deterministic rigid-rotation test-time
augmentation was implemented as a validation-only candidate. It rotates the raw
cached geometry before the checkpoint's normal per-graph preprocessing, averages
class probabilities across identity plus fixed seeded SO(3) rotations, and records
the matrices in the evaluation JSON. This avoids the incorrect shortcut of rotating
already channel-standardized tensors. Targeted augmentation/evaluation regression
tests pass (18 tests); its validation effect has not yet been measured and it will
not be applied to test unless selected without test feedback.

Checkpoint probability ensembling is also implemented for validation screening.
Each member is reconstructed from its own embedded configuration (so auxiliary-head
architectures remain loadable), while task, feature dimensions and class count are
checked for compatibility. Member probabilities are averaged before argmax and the
exact checkpoint paths/epochs are recorded. This will test whether the canonical,
always-rotated, mixed-rotation and operation-supervised models have complementary
errors; test remains untouched during ensemble selection.

The validation search is now queued as a reproducible GPU-4 job after both round-4
training results. It snapshots its evaluation source, records hashes, evaluates six
predeclared candidates (four-checkpoint combinations are not added after seeing
results), sorts by face accuracy, and writes `validation_search_v1/summary.json`.
The search manifest explicitly records `selection_split: val` and
`test_accessed: false`.

An independent `context_rotation_seed43` 100-epoch fine-tune is queued after the
mixed-rotation run, again reusing the same frozen 50-epoch context encoder. This
tests seed robustness and adds a genuinely independent member for ensembling.
The earlier validation-search wait was stopped before it used GPU time to prevent
it racing this training job; validation search v2 waits for seed 43 as an explicit
dependency and adds two- and multi-seed ensembles. GPU 4 remains serial.

Validation search v2 also predeclares a temporal ensemble between the rotation
run's validation-best checkpoint and its independently saved epoch-100 snapshot.
This probes complementary late-training errors without changing the prescribed
100-epoch budget or launching another training run. It remains validation-only.

A confidence-gated topology postprocessor is implemented for a later validation
ablation. It averages incoming adjacent-face probabilities, but scales the update
by `(1 - face confidence)`, leaving probability-one predictions and isolated faces
unchanged. The alpha is range-checked and recorded in evaluation JSON. This targets
locally inconsistent uncertain predictions without indiscriminately blurring
high-confidence operation boundaries. It is not part of the already snapshotted v3
search and has not accessed test data. Targeted tests pass (21 tests).

## Round 4 operation result

`context_rotation_operation` completed all 100 fine-tuning epochs. Validation-only
selection chose epoch 97 with **96.9082%** face accuracy over 79,920 faces. This is
only +0.0025 percentage points over `context_rotation` (96.9057%), so the auxiliary
operation-family task is effectively neutral for aggregate accuracy, although its
macro F1 is 0.9223. The manifest still records `test_accessed: false`. The queued
`context_rotation_mix` run then started automatically on GPU 4.

## Round 4 mixed-rotation result

`context_rotation_mix` completed all 100 fine-tuning epochs. Validation-only
selection chose epoch 90 with **97.2410%** face accuracy over 79,920 faces and
macro F1 0.9242. This is +0.3353 percentage points over the always-rotated model
and is the first candidate above the 97% validation threshold. The checkpoint is
now frozen for one independent official-test evaluation. Its result and manifest
still record `test_accessed: false`; test evaluation is deferred until the already
started seed-43 run releases GPU 4. The waiting validation-search job was stopped
before GPU use so the frozen test has exclusive priority next.

## Final 97% result

The single mixed-rotation checkpoint scored **96.9197%** on the official test split,
so it did not itself meet the target. After seed 43 completed, the predeclared
validation-only search evaluated its fixed candidate list. The best candidate was
the equal-probability ensemble of `context_rotation` epoch 92,
`context_rotation_mix` epoch 90, and `context_rotation_operation` epoch 97, with
**97.4087% validation accuracy**. No test predictions were used to choose its
members or weights.

That frozen ensemble was then evaluated once on the official Fusion360Seg test
split and achieved **97.0974% face accuracy** (74,833 / 77,070 correctly labeled
faces across 5,366 CADs), exceeding the requested 97% target. Weighted F1 is
0.9708 and macro F1 is 0.9096. All three members reuse the same Fusion360Seg-only
50-epoch pretraining run and each completed exactly 100 Fusion360Seg-only
fine-tuning epochs. Their manifests share identical disjoint split hashes and
record GPU 4. `validation_search_v4/summary.json` now records the selected test
artifact and `test_accessed: true`; no further experiments were started.

## Code changes and gains over baseline

The implementation work was concentrated in the following modules:

- `models/encoder.py`: added optional face/edge grid encoders and learned
  multi-scale context pooling. The latter mixes the input and intermediate GNN
  layers, then broadcasts per-CAD mean/max context back to each face.
- `models/diffusion.py`: added consistent categorical attribute masking during
  pretraining, including shared masks for reverse/parallel edge records and loss
  computation only at masked positions.
- `models/segmentation.py` and `models/downstream.py`: added boundary auxiliary
  supervision, boundary-gated refinement, operation-family auxiliary supervision,
  and structured graph-label diffusion support. Ordinary inference continues to
  emit the original eight Fusion360Seg classes.
- `data/graph.py` and `data/dataset.py`: added physical/isotropic geometry
  normalization, rigid SO(3) rotation, UV-grid symmetries, deterministic evaluation
  rotations, and a configurable rotation probability. The winning mixed policy
  retains the canonical frame for 50% of training CADs and rotates the other 50%.
- `brep/occ_extractor.py`: added optional local shared-edge normals derived from
  original STEP pcurves, stored in a separate cache without overwriting baseline
  features.
- `training/finetune.py` and `training/pretrain.py`: added scheduler support,
  auxiliary-loss handling, and progress-output control while preserving the exact
  50/100 epoch protocol.
- `training/evaluate.py`: added deterministic rotation TTA, compatible multi-model
  probability ensembling, and confidence-gated topology smoothing. Evaluation JSON
  records member checkpoints, epochs, transforms, and postprocessing parameters.
- `scripts/fusion97_*.py`: added split auditing, source snapshots/hashes, strict
  GPU-4 experiment launching, local-edge cache preparation, status reporting, and
  fixed validation-only candidate searches. Regression coverage was expanded in
  `tests/test_fusion97_variants.py` and `tests/test_evaluate.py`; the final suite
  passes 144 tests.

All comparisons below use the same 79,920-face validation split. The baseline is
**96.1974%**, so “gain” is an absolute percentage-point difference, not a relative
percentage. Every trained row completed 50 Fusion360Seg-only pretraining epochs
(or reused the stated frozen 50-epoch encoder) and 100 Fusion360Seg-only fine-tuning
epochs.

| Variant | Main code/experiment change | Validation accuracy (%) | Gain vs baseline (pp) |
|---|---|---:|---:|
| baseline | 128-dim, four-layer edge-update attention | 96.1974 | +0.0000 |
| masked | 50% categorical masking in pretraining | 96.0786 | -0.1189 |
| wide | Hidden dimension 128 → 256 | 96.2588 | +0.0613 |
| grid | Dedicated face/edge grid encoders | 96.0861 | -0.1114 |
| context | Learned multi-layer + graph mean/max context | 96.3889 | +0.1914 |
| boundary | Boundary auxiliary loss | 96.2112 | +0.0138 |
| boundary_refine | Boundary loss plus gated neighbor refinement | 96.1324 | -0.0651 |
| geometric | Isotropic physical normalization | 95.6619 | -0.5355 |
| geometric_grid | Physical normalization plus grid encoders | 95.9735 | -0.2240 |
| localedge | STEP-pcurve local shared-edge normals | 96.1699 | -0.0275 |
| context_cosine | Context plus cosine LR decay | 96.2400 | +0.0425 |
| context_operation | Context plus operation-family auxiliary loss | 96.2988 | +0.1014 |
| context_rotation | Context plus 100% random rotation | 96.9057 | +0.7082 |
| context_rotation_seed43 | Independent seed-43 rotation run | 96.8143 | +0.6169 |
| context_rotation_operation | Rotation plus operation auxiliary loss | 96.9082 | +0.7107 |
| context_rotation_mix | 50% canonical / 50% random rotation | 97.2410 | +1.0435 |
| final ensemble | Rotation + mixed rotation + rotation/operation | **97.4087** | **+1.2112** |

The largest single-model gain came from mixed rotation, not additional capacity or
geometry normalization. Relative to always rotating every training CAD, retaining
50% canonical samples added another **0.3353 pp**. Equal-probability ensembling of
three validation-selected models added **0.1677 pp** over the best single model.
The final official-test accuracy is **97.0974%**. A baseline test evaluation was
intentionally never run, so the report does not claim a test-set gain versus
baseline; doing so would require an additional baseline test access not used in
the validation-driven protocol.
