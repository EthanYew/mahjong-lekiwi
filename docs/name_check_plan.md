# A second tile-name read from the wrist camera: LLM, VLM or classifier

Written 2026-11-03, first test run 2026-11-04. A proposal: nothing here runs on the arm. It grew out of Move 3 in
`docs/retrospectives/2026-11-02_week3_skills_assessment.md`. What's done and next stays in ROADMAP.md's status table.

## The question

After YOLO (the table-camera detector) reads the tiles, a person settles the doubtful names. Could a model do that
instead, and which kind: an LLM (text only), or a VLM (vision-language model, which also takes images)? A draft pipeline
proposed fine-tuning Qwen2.5-VL-3B on 800+ labelled tile images and running it on a Raspberry Pi 5.

## The answer so far

- **VLM over LLM.** An LLM can't see the tile. All it would get is YOLO's reads and the game state, and the repo already
  handles those with rules (`identity.Votes`, `game.too_many`).
- **Test a small classifier and untrained VLMs before fine-tuning anything.** Choosing 1 of 34 faces from one camera is
  what a classifier is built for, and it already has a place in ROADMAP 10 and `docs/autonomy_plan.md` 8.5. This is an
  expectation, not a measurement. The frames below make a VLM more plausible than first assumed, because many tiles carry
  readable marks, so both get tested.
- **The reader's job is the name, not presence.** Whether a tile is there now has its own answer: the detector's faint
  boxes (`perception.faint_tiles`), then the wrist look if those don't settle it. The last 58 attempts on the mat logged
  77 `tile_faint` events and no looks, so the look's frame isn't the place to read. An attempt's first frame at the hover
  is (`wrist_0000.jpg`), just before the arm commits. A reader's "no tile" answer is there so it doesn't invent a name for
  an empty spot.
- **Enough photos exist to start, with no arm.** All 34 names are in the 1,004 usable photos, though mostly on the black
  top. Whether that training carries over to the mat decides how much mat data is needed, and it can be tested today.
- **Read every name, not only low-confidence ones.** Wrong names aren't reliably low-confidence, and the votes can accept
  one (below).
- **Not decided:** Move 3 says tile names should "only make the robot ask more often, never to guess". This plan keeps
  that: agreement accepts a name, and disagreement still asks. Removing the person from disagreements too would change
  that rule, and that's the owner's call.

## Evidence

All of it is small. Together it says where to look, and proves nothing about how often a reader would be right.

### YOLO's confidence on the old exam
The live detector on the labelled exam (`20261027_172928_capture`: 15 frames, 126 reads, tiles repeating across frames),
with boxes kept down to 0.25. The totals match `data/detector_eval/tile_detector@0c4b15eb.json`.

| Read confidence | Right | Wrong |
|---|---|---|
| Below 0.7 | 8 | 1 (2C read as 3C, at 0.44) |
| 0.7 and above | 114 | 3 (flowers, at 0.89-0.92) |

So a low-confidence trigger would have missed 3 of the 4 wrong names, and 8 of the 9 reads it did catch were already right.
Play excludes flowers. This exam predates the standard mat, and `20261028_162624_capture`, the second session listed in
`data/detector_exam.json`, is in no stored score.

### The wrist look (`mahjong/wrist_check.py`)
It runs when a scan loses a tile and asks whether a tile face is between the jaw tips. It doesn't read names.

**Dark surfaces, 15 looks, all 2026-10-30** (`20261030_115706_bench`, `20261030_171411_bench`, `20261030_174042_bench`,
`20261030_175235_bench`), graded against what happened next:

| Verdict | Looks | Later scans or pickup agree | Only the saved frame agrees | Wrong |
|---|---|---|---|---|
| Still there | 8 | 6 | 1 (the session ended) | 1 |
| Not there | 7 | 5 | 2 | 0 |

The one miss (12:06, 1B) was a neighbour: a knock pushed 1B 36mm away and 3D to 20mm from the spot, and the look saw 3D's
face. That led to `wrist_check.CROWDED_M`. 5 of the 8 "still there" looks were for 1B, and 3 of the 15 grades rest on
viewing the frame.

**Standard mat, 11 looks** (`20261103_121112_bench`): all saw a tile, but this doesn't validate the test, because bare white
stripe also reads 0.72 against the 0.5 cutoff. Faint boxes, which answer all 11 offline, have since replaced these looks on
the arm. See `docs/invariant_notes.md` for those and for how the look handles a pale surface.

### YOLO's votes on the standard mat
Reads from `20261103_121112_bench` (74 scans) replayed through `identity.Votes`, grouped by place:

| Tile (from the frames) | What YOLO said | Vote result |
|---|---|---|
| 2B at S7 | 4B in 48 scans (median confidence 0.71), 3B in 7, never 2B | **Accepts 4B without asking** (0.875 share, bar is 0.8) |
| 1B at S6 | 1B in 46 scans, EW in 21 (median 0.87) | Asks a person (0.70 share) |

Caveats: this is a replay of a bench session, not a hand; it pools a whole session's scans; and logged scans hold each
tile's best read, not per-frame votes. Even so, a wrong name clearing the bar is what `identity.py`'s first-guess bars need
checking against before an attended hand. `docs/invariant_notes.md` reaches the same conclusion from single frames: 2B reads
as 4B in 45 of 46, so no vote fixes it and names need retraining.

### Can the frames be read at all?
The saved wrist frames (`sessions/*/wrist_checks/`) are sharp. Printed corner marks and characters are clear, on the black
top and on the stripes. A person or a large VLM can see 2B, 1B and 3C where the logs say otherwise, and 3D in the 12:06
miss. In 4 of the 7 "not there" frames, the missing tile is visible elsewhere in view. This was an informal check by Claude,
which had read the logs and seen the expected names, and is a much larger model than Qwen 3B. It shows the input is good
enough, and nothing about any small model.

### The photos we already have
From `wrist_photos.labelled`: each `wrist_capture.py` view and each bench attempt's first frame at the hover.

| Surface | Usable photos | Names, of 34 | Reviewed before this audit | Notes |
|---|---|---|---|---|
| Black top | 858 (488 bench, 370 `wrist_capture`) | 34 | 509 | 12 names rest on 1-2 placements: 1C, 2C, 3B, 4B, 4C, 5B, 7C, 8B, 8C, 9B, 9C, RD |
| Standard mat | 87 (bench) | 26 once corrected | 0 | 2-6 photos a name, mostly near-duplicate pairs |
| Green mat, other | 59 (bench) | 14 each | 0 | |

The 56 mat demos and about 10 mat eval episodes also hold hover frames, inside dataset videos. Their logged names look right,
but that's unchecked.

**How often the logged names are wrong.** A person fixes names at `wrist_capture.py` but not at the bench, where a photo
takes YOLO's read, or the spot's ID if YOLO found no name. Claude viewed all 495 unreviewed bench photos:

| Photos | Wrong name | Not usable |
|---|---|---|
| 87 on the standard mat | 11 (1B, 2B, 3C, 8C, 2C) | 2 |
| 408 elsewhere | 4 (1B as EW, 5C as 6C) | 2 |

So logged names are mostly right off the mat and often wrong on it. Those 11 come from 5 physical tiles, so this is a few tiles
misread repeatedly, not wrong names spread evenly. It also changes what's missing: logged counts show 7C and EW on the mat and no 2B or 8C, but it's the reverse. Never yet on
the mat: **2D, 4B, 6D, 7B, 7C, 9B, 9C, EW**. The 19 proposed verdicts, a sheet of them and a script to record them are in
`docs/retrospectives/evidence/2026-11-03_name_check_audit/`. They are Claude's reading: only the owner's check makes them
verdicts, and the 476 photos that looked right are still unreviewed.

On the mat, the tile is often well off `wrist_check.EXPECTED_PX`, sometimes with that point at its edge, so a crop has to be
larger than the disc.

## First test, 2026-11-04: step 2 of the plan

**A small classifier far outread untrained VLMs, and fine-tuning a VLM isn't justified yet. The classifier catches YOLO's wrong
names, but it asks a person too often to remove the person.** This is a screen on 154 mat photos, not an exam.

Trained on 830 non-mat photos, tested on all 154 usable mat photos (55 placements, all 34 names, labels reviewed by the owner).
All in `docs/retrospectives/evidence/2026-11-04_name_check_test/`.

| Reader | Right | Agrees with YOLO and wrong | Would ask a person | Seconds a photo here |
|---|---|---|---|---|
| YOLO's own read | 135 (88%) | | | |
| Classifier, 240px crop (3 seeds) | 128-140 (83-91%) | 0, 0, 1 | 21-27% | under 0.01 |
| Classifier, 360px crop (3 seeds) | 90-119 (58-77%) | 2, 8, 7 | 25-44% | under 0.01 |
| Moondream2 | 50 (32%) | 0 | 68% | 0.23 |
| Qwen2.5-VL-3B, 1024 image tokens | 41 (27%) | 0 | 74% | 0.44 |
| SmolVLM-500M | 34 (22%) | 0 | 79% | 0.04 |

What it shows:
- **The crop matters more than anything else tried.** The tighter crop gained 9-45 photos per seed. Neighbours in a wider crop
  are the likely reason, not yet shown.
- **The classifier disagrees where YOLO is wrong.** It flagged 56 of YOLO's 57 wrong reads over 3 seeds. The one it missed is the
  table's one wrong acceptance: a 9C both called 5C, at 0.33 confidence, so a confidence bar would also have caught it.
- **That is safety, not autonomy.** Bench photos, the closest to real use, were 94% right with 19% asks, about 5 asks over a hand's
  27 reads. Confidence doesn't fix that yet: at 0.9 no name was wrong, but only 25% of reads cleared it, and 23% once agreeing with YOLO too.
- **The weak spots are thin names.** Of 59 wrong answers over 3 seeds, 7C and 4B were 39 (21 and 18), mostly 7C as 6C. They have
  14 and 10 non-mat photos, from 2 and 1 placements. Capture views were 78% right, against bench photos' 94%; they vary more, but the
  two also cover different tiles, so this doesn't separate the causes.
- **Mat photos in training helped little.** Holding out each mat tile set in turn, adding the other two moved it from 100% to
  100%, 90% to 92%, and 79% to 84%. Seeds alone vary by more, and there are few mat photos per name.
- **Untrained VLMs mostly confuse suits** (9C as 9B, 4B as 4D, 7C as 7B) and overuse a few answers. Qwen does see the image: in free
  text it read some marks like 3 三, E 東 and corner C and P, and got others wrong (`vlm_check.json`). Fine-tuning could do better,
  which this doesn't rule out, but a classifier costs seconds to train where that costs far more.

What it can't show: other lighting (every mat photo has the lamp and ceiling lights on), speed on a Pi, or the ~1000-read gate.
Photos of a placement aren't independent. The 240px crop was fixed in advance, but seeing it win picks it, so a real exam has to
confirm it. Each VLM used one prompt, unchanged, and two got a settings fix after their first pass: Qwen for image tokens (it
scored 14% before), and Moondream's chat template, without which it failed. No reader was given a "no tile" photo. Seeds differ by up to
12 photos, so small gaps mean little.

**What's next:** give 7C, 4B and the other thin names more placements, measure tile offsets for the crop, and retest. Only then
is a real exam (steps 3-4) worth recording. Whether YOLO needs retraining is step 5's question.

## The plan

Training happens on the RTX 5080 machine. Inference happens there now, and on a Raspberry Pi 5 (8 GB) later, for play
testing in other places.

1. **Review the names** (no arm). The owner checks the audit's 19 proposed verdicts, then looks through the rest in
   `scripts/wrist_review.py`.
2. **First test: does black-top training carry to the mat?** (No arm.)
   - Train the small classifier on the reviewed non-mat photos, and test it on the 85 usable mat photos.
   - If a crop that's mostly tile face carries over, the mat needs only a top-up. If it fails, the mat needs real training
     data before anything else.
   - Run the stock VLMs on the same photos. This is a screen, with 26 names and near-duplicates, not an exam.
   - Report per physical tile, and in particular 1B, 2B, 3C, 8C and 2C, which the bench logged wrongly.
3. **Grow mat data through normal bench work.**
   - Swap the 8 never-on-mat names into the next `even-hand14` bench sessions. Each session gives 14 new placements, and
     correcting a bench photo's name is one review.
   - Use `wrist_capture.py` only for what's left: the 12 thin names, and lighting. Correct names on screen before accepting,
     since on the mat the proposed names come from YOLO.
   - Mat photos so far all have the lamp and ceiling lights on, in dim or no daylight. The mat travels with the setup and the
     room's light doesn't, so daylight and lamp-off sessions are the real gap.
4. **Hold a real exam.** Whole sessions in lighting chosen in advance, on the standard mat, with all 34 names, listed in a new
   `data/name_check_exam.json` (following `data/detector_exam.json`) and never trained on. Today's mat sessions can't be it:
   two tile sets, one lighting family, and step 2 will have looked at them. Unlike ACT's offline scores
   (`docs/retrospectives/2026-11-03_eval_validity.md`), a name exam measures the task itself, if its frames match how the
   reader will be used.
5. **Score YOLO on the mat exam before deciding to retrain it.** It has never had a mat exam, but it has trained on the mat:
   about 536 of the 1,310 frames in `data/detector/train_v6x3` show the striped mat (a colour check on frame centres), and it
   still reads 2B as 4B. So more mat frames may not fix its names, especially with a tile only ~29px wide in that view. If the
   exam's mix-ups point to retraining, bench scans on the mat are saved for `mine_frames.py`. A retrain changes which errors a
   second read has to catch, so score second reads against the YOLO that will actually run.
6. **Compare the readers on that exam.**

   | Reader | Setup | Why |
   |---|---|---|
   | Small classifier (YOLO-cls nano) | Train here; NCNN export for the Pi | The repo's existing idea, and most likely to fit the Pi |
   | Small stock VLM | SmolVLM (256M or 500M) or Moondream2, GGUF, no training | The sizes most likely to be usable on a Pi |
   | Qwen2.5-VL-3B, stock | `ggml-org` Q4_K_M + mmproj, no training | The draft's model, about 3 GB of files |
   | Hosted VLM (optional) | Needs a key and a budget | A laptop benchmark for how good a VLM can get; venues may have no network |

   Every reader gets the same crop, sized from a measurement of how far tiles sit from `wrist_check.EXPECTED_PX`. A larger
   crop admits neighbours, so training and testing must match. It answers from all 34 names or "no tile", **and is never
   given YOLO's candidates or the expected name**: YOLO's options for the 2B tile were 4B and 3B.
7. **Score them the same way** (a new `scripts/name_check_eval.py`, reusing `detector_eval.wrong_rate_bound`).
   - The gate: names accepted and wrong, when paired with YOLO's vote. It must be 0, towards the ~1000 reads in
     `docs/autonomy_plan.md`. Report distinct tiles and sessions too, as photos of one tile aren't independent.
   - Also: share of asks removed, whether it says "no tile" on an empty spot, a usable confidence, seconds per read, memory.
8. **Time the finalists on a Pi 5, with the whole stack.** A hand has about 27 name reads, so 30 seconds a read (an
   illustration, not a measurement) would add about 13 minutes. ACT runs at 30 Hz on the GPU today and YOLO at 1280 px. If
   ACT can't keep up on a Pi, that decides the setup before any reader does.
9. **Decide.**
   - A reader passes: wire it in as a second read on every name entering the hand and every drawn tile. Agreement accepts;
     disagreement asks. Compare with the slot's established name, not with a lost detection's name. Don't use
     `Votes.known()` for a model's answer: it's a person's word, and later scans can't outvote it. Log every read as an
     event. An attempt's first frame covers the tile about to be picked; reading the 13-tile hand at the start needs its
     own hover over each tile, which needs timing on the arm.
   - Only a stock VLM comes close: fine-tune, with the corrections below.
   - None pass: keep asking a person, and write that up. It's a real finding.

Before calling a reader ready, smoke-test it on the 11 frames in `20261103_121112_bench/wrist_checks/`: it should say 2B, 1B
and 3C, not the logged names. A pass on the exam only screens. A real hand on the mat with the second read on, graded by
`interventions.hand_grade`, decides.

## If fine-tuning is ever justified: what to fix in the draft

The draft, in short: convert images to Qwen's chat format, fine-tune with 4-bit QLoRA through TRL, merge the adapter, convert
to a quantized GGUF, and call it from `llama_cpp` on the Pi when YOLO is unsure.

Terms: *fine-tuning* is extra training that adapts an existing model to a task. *LoRA* does that by training a small add-on
(an *adapter*). *GGUF* is the file format llama.cpp runs. *Quantizing* stores a model's numbers less precisely, so it's
smaller and faster, with some loss of quality. *mmproj* is the separate file holding the model's vision part.

| # | What the draft does | What goes wrong | What to do |
|---|---|---|---|
| 1 | Expects a folder of cropped photos per tile, named like `7-character` | The repo's labelled frames are whole-table photos, and it names tiles `7C` | Use wrist photos and the repo's names |
| 2 | Trains on every photo | Nothing is left to test on, so accuracy is unknown | Hold out whole sessions |
| 3 | Asks "identify this tile" in training, then "is it one of these?" in use | It's tested on a question it never practised, and the list can omit the right answer | One question throughout, with all 34 names |
| 4 | Relies on `remove_unused_columns=False` to keep the images | Nothing actually loads and batches the images, so training likely fails or learns nothing useful | Add a collator (or confirm the pinned TRL version handles it), and prove it on one batch |
| 5 | Says it trains the language part, but `gate_proj`, `up_proj` and `down_proj` also match vision layers | You wouldn't know what you changed, and vision changes must reach the mmproj | Choose on purpose, and build the mmproj from the fine-tuned model |
| 6 | Runs `convert_hf_to_gguf.py --outtype q4_k_m` | That value is rejected, the clone URL is cut off, and no mmproj is made | Convert to f16, compress with `llama-quantize`, and export with `--mmproj` |
| 7 | Uses `MoondreamChatHandler`, no mmproj, and builds `Llama()` on every read | It can't see the image, and each read waits for a multi-GB load | Use `Qwen25VLChatHandler` with the mmproj, and load once at startup |
| 8 | Returns whatever text the model writes | It can't say "not sure", so a wrong answer becomes a silent wrong tile | Limit answers to legal names, check how sure it was, and ask a person when unsure |
| 9 | Installs `transformers`, `peft`, `trl`, `bitsandbytes` alongside everything else | They aren't in the `lerobot` environment and could disturb it | Use a separate training environment |

The draft's sizing holds up: the published Q4_K_M model is 1.93 GB, plus 0.85-1.34 GB for the mmproj, against the Pi's 8 GB.
What's unknown is speed: no published CPU timing for this model on a Pi turned up.

## Sources

- [transformers: Qwen2.5-VL model code](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py) (layer names)
- [llama.cpp: `convert_hf_to_gguf.py`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py) (`--outtype` values, `--mmproj`)
- [llama.cpp: multimodal docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multimodal.md) (supported models)
- [ggml-org: Qwen2.5-VL-3B GGUF files](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/tree/main) (file sizes)
- [llama-cpp-python README](https://raw.githubusercontent.com/abetlen/llama-cpp-python/main/README.md) (chat handlers)
- [TRL: VLM example](https://huggingface.co/spaces/ivangabriele/trl-sandbox/blame/main/examples/scripts/sft_vlm.py) (custom collator)
- [Ultralytics: Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) (about 67 ms per image for its nano detector with NCNN at 640 px; our models aren't timed)
