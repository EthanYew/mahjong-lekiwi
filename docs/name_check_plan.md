# A second tile-name read from the wrist camera: LLM, VLM or classifier

Written 2026-11-03, against `main` at `98170b0`. A proposal: nothing here is built. It grew out of Move 3 in
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
- **Pair the reader with the wrist look; don't simply replace it.** The look's colour test still works on dark surfaces.
  On the pale standard mat, the reader's "tile / no tile" answer could take over the first-look question to a person.
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
stripe also reads 0.72 against the 0.5 cutoff. See `docs/invariant_notes.md` for how the look handles that now.

### YOLO's votes on the standard mat
Reads from `20261103_121112_bench` (74 scans) replayed through `identity.Votes`, grouped by place:

| Tile (from the frames) | What YOLO said | Vote result |
|---|---|---|
| 2B at S7 | 4B in 48 scans (median confidence 0.71), 3B in 7, never 2B | **Accepts 4B without asking** (0.875 share, bar is 0.8) |
| 1B at S6 | 1B in 46 scans, EW in 21 (median 0.87) | Asks a person (0.70 share) |

Caveats: this is a replay of a bench session, not a hand; it pools a whole session's scans; and logged scans hold each
tile's best read, not per-frame votes. Even so, a wrong name clearing the bar is what `identity.py`'s first-guess bars need
checking against before an attended hand.

### Can the frames be read at all?
The saved wrist frames (`sessions/*/wrist_checks/`) are sharp. Printed corner marks and characters are clear, on the black
top and on the stripes. A person or a large VLM can see 2B, 1B and 3C where the logs say otherwise, and 3D in the 12:06
miss. In 4 of the 7 "not there" frames, the missing tile is visible elsewhere in view. This was an informal check by Claude,
which had read the logs and seen the expected names, and is a much larger model than Qwen 3B. It shows the input is good
enough, and nothing about any small model.

## The plan

Training happens on the RTX 5080 machine. Inference happens there now, and on a Raspberry Pi 5 (8 GB) later, for play
testing in other places.

1. **Capture on the standard mat** (needs the arm).
   - `wrist_capture.py`, all 34 playable faces, named by the person placing them, never by YOLO.
   - Also capture the bare mat at the disc, so a reader can learn "no tile".
   - Hold out whole sessions in other lighting as the exam (a new `data/name_check_exam.json`, following
     `data/detector_exam.json`). The mat travels with the setup; the room's light doesn't.
   - Save scans too: the same sessions give YOLO its first exam on this mat.
   - Train on these plus the 744 older photos (`mahjong/wrist_photos.py`, `data/wrist_review.jsonl`). Those are all from
     2026-10-28 on the black top, with 4-9 names per session, so they can't be the exam.
2. **Retrain YOLO first if the mat exam calls for it.** It may be the cheapest fix, and it changes which errors a second
   read has to catch.
3. **Compare the readers on one exam.**

   | Reader | Setup | Why |
   |---|---|---|
   | Small classifier (YOLO-cls nano) | Train here; NCNN export for the Pi | The repo's existing idea, and most likely to fit the Pi |
   | Small stock VLM | SmolVLM (256M or 500M) or Moondream2, GGUF, no training | The sizes most likely to be usable on a Pi |
   | Qwen2.5-VL-3B, stock | `ggml-org` Q4_K_M + mmproj, no training | The draft's model, about 3 GB of files |
   | Hosted VLM (optional) | Needs a key and a budget | A laptop benchmark for how good a VLM can get; venues may have no network |

   Every reader gets a crop around the disc (`wrist_check.EXPECTED_PX`), since neighbours are in view. It answers from all
   34 names or "no tile", **and is never given YOLO's candidates or the expected name**: YOLO's options for the 2B tile
   were 4B and 3B.
4. **Score them the same way** (a new `scripts/name_check_eval.py`, reusing `detector_eval.wrong_rate_bound`).
   - The gate: names accepted and wrong, when paired with YOLO's vote. It must be 0, towards the ~1000 reads in
     `docs/autonomy_plan.md`. Report distinct tiles and sessions too, as photos of one tile aren't independent.
   - Also: share of asks removed, whether "no tile" works on the pale mat, a usable confidence, seconds per read, memory.
5. **Time the finalists on a Pi 5, with the whole stack.** A hand has about 27 name reads, so 30 seconds a read (an
   illustration, not a measurement) would add about 13 minutes. ACT runs at 30 Hz on the GPU today and YOLO at 1280 px. If
   ACT can't keep up on a Pi, that decides the setup before any reader does.
6. **Decide.**
   - A reader passes: wire it in as a second read on every name entering the hand and every drawn tile, and during the
     wrist look. Agreement accepts; disagreement asks. Compare with the slot's established name, not with a lost
     detection's name. Don't use `Votes.known()` for a model's answer: it's a person's word, and later scans can't outvote
     it. Log every read as an event.
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
