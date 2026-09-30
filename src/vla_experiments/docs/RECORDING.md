# Recording demonstrations — contract, operator guide, validation, pilot

Since 2026-09-30 (raw format `morty_vla_raw_v3`, command events `morty.command_event.v1`).
The machine-readable contract (states, segment codes, outcomes, the command
event fields, clock domains, the observation/action boundary, compound
commands) is the docstring of `vla_experiments/recording_contract.py`.
This file is for the people who record, check and export.

## 1. What changed and why

Up to v2, an instruction was treated as active as soon as its intent arrived on
`/robot/intent`. But the intent router queues, deduplicates, defers and refuses
intents, and it also answers questions with them. So an intent arriving is not
proof that anything was executed:

| trial | what the v2 labels said | what happened |
|---|---|---|
| 114 | `go_home` labelled the `look_right` motion (the two intents arrived 8 ms apart), `look_right` got 0 steps | the router queued go_home behind the look |
| 121 | `release_home` (said before a pointed pick) labelled that pick's approach, close and lift | the answer waited for the release pause |
| 126 | "open and go home" became just "open the gripper" | the `then` part was dropped |

From v3 on:

- **Router.** It publishes each command's lifecycle on `/robot/command_event`
  (received → queued/deferred → accepted → cycle → completed / failed /
  rejected / cancelled / superseded).
- **Command ids.** Every intent carries a `command_id` and the `utterance_id`
  it came from. It also carries the question it answers, if any, and its
  place in a multi-step plan (`plan_id` / `plan_index`).
- **Executor task state.** The executor's task state (approaching, carrying,
  awaiting_release, releasing, returning_home …) is recorded **as it
  happens**, not only when the cycle ends.
- **Step labels.** Steps are labelled from the command that is **executing**
  (`labels.py`). When the evidence for an interval is missing, the segment
  is flagged for review instead of being invented.

### Segments (`steps.npz["segment"]`)

| code | name | meaning | exported by default |
|---|---|---|---|
| 0 | idle | setup, conversation, nothing executing | no |
| 1 | task | the accepted command executing, including its meaningful holds | **yes** |
| 2 | reset | automatic cleanup after a failure / stop, an automatic close cut | no |
| 3 | hold | the robot waits at a question pause for the next words | no |
| 4 | clarify | a which-one / pointing question is open | no |
| 5 | assist | the operator marked manual help (✋ assist) | no, never as autonomous success |

**Compound commands stay one instruction** across what was executed:

- "open the gripper and go back home" covers the release **and** the return;
- "bring it home and hand it over" covers the carry, the hold and the release.

If only part of a compound was executed, the instruction is downgraded to the
part that ran (`_executed_form`). The robot is never taught the desired action
when it did something else.

### Outcomes and quality

- **Trial outcomes:** `success | failure | unknown | interrupted`.
  - A bare `stop` or an unrecognised argument (e.g. `stop sucess`) gives
    **unknown**. It never means success.
  - Stopping while a command is still running marks that command
    **interrupted**.
- **Commands and segments** keep their own outcomes in `meta["commands"]`.
- **`meta["quality"]`** has three separate parts:
  - `integrity`: files, images written, steps;
  - `alignment`: camera skew, clock offset, stale gripper, repeated frames;
  - `semantic`: `label_source`, `verified`, `review_required`, `flags`, `notes`.
- **`training_eligible`** = integrity ok **and** alignment ok **and** labels
  verified by execution evidence.
- **Flags vs notes** (since the 30 Sep review). A *flag* needs a human look
  and makes the labels unverified. A *note* records something the evidence
  already settles:
  - the executor re-closing the gripper after a completed release, with the
    arm standing still (labelled as cleanup);
  - a pick the executor reported **failed before it moved** (nothing to label);
  - single steps left out for timing (see Alignment).

A green health flag is **not** proof that the demonstration is right. Look at
the trial before you use it.

### Alignment

- **Images.** Step *i* takes, for each camera, the frame whose **source** stamp
  is closest to the tick time. Each step records:
  - the frame's age (`img_src_age_*`);
  - whether the frame repeats the previous one (`img_repeat_*`);
  - the skew between the two cameras (`camera_skew`).
- **Robot state.** The newest sample that is not newer than the tick
  (`gripper_age` for the gripper).
- **Astra clock offset.** The Astra runs on the shuttle's clock. Its offset
  is measured per episode from new frames only and stored in
  `meta["alignment"]`.
- **Tolerances** (recorder parameters, `config/recorder.yaml`):

  | parameter | default | what it bounds |
  |---|---|---|
  | `max_camera_skew_s` | 0.35 | skew between the two cameras |
  | `max_gripper_age_s` | 2.0 | age of the gripper sample |
  | `max_clock_offset_s` | 0.5 | the Astra clock offset |
  | `max_image_age_s` | 0.5 | age of a camera frame (receipt, and capture when the clocks are comparable) |
  | `save_timeout_s` | 20 | image writes must finish within this at stop |

- **Per step, not per episode** (since the 30 Sep review). A step outside
  `max_image_age_s`, `max_camera_skew_s` or `max_gripper_age_s` is left out
  of training (`raw_dataset.step_misaligned`); the episode stays eligible.
  Blocking are only episode-wide problems: step gaps, dropped ticks, a frozen
  stream, clocks that are not comparable, gripper lag, or bad steps in more
  than 10 % of the task steps (at least 4). `quality.alignment.details` keeps
  every check's wording, and `issues` holds only what blocks.

- **Actions.** An action is the change to the next step **within the same
  segment**. The last step of a segment has zero motion. No action crosses
  a command change or an excluded gap (`derive_actions(ep, sel)`).

Units, frames and gripper convention are as in README.md ("Raw format"):
joints in rad, TCP in `base_link` in m + quaternion, gripper 0 = closed / 1 = open (from the gripper joint; exported `gripper_target` is the same, binary).

### Files of a saved episode

| file | content |
|---|---|
| `recording_manifest.json` | written at start (`state: recording`) and **last** at save (`complete` / `incomplete`), with `run_id`, step and image counts, integrity issues. A folder without `state: complete` is not a finished recording. |
| `meta.json`, `steps.npz` | written atomically (tmp + rename) |
| `images/` | the image writer is drained (queue empty **and** no write in progress); failed writes are listed |
| `vla_previous/<time>_<run>/` | a previous episode in a **reused** trial folder is moved here, never overwritten |

### Bag

`BAG_TOPICS` (30 topics) now includes:

- `/robot/command_event`
- `/grasp_executor_lin_llm/task_state`
- `/gripper_controller/command_event`
- `/grasp_pipeline/point_answer`
- `/robot/open_question`

The GUI writes `bag/bag_status.json`, which holds the recorder's exit code and
the stop stage (1 = SIGINT, 2 = SIGTERM, 3 = SIGKILL). `bag_check.verify_bag`
checks:

- `metadata.yaml`;
- that the database exists and is not a Git LFS pointer;
- sqlite `quick_check`;
- message counts, required topics and the time window;
- a clean exit. A recorder killed only **after** its log says "Recording
  stopped", with every check above passing and nothing missing at the end,
  is a *warning* ("the data is complete"), not a failure. That happened in
  trials 140 and 146: rosbag2 hung at exit for 11 s.

Post-processing checks the bag only after `bag_status.json` says
`state: exited`. The file exists from the start of the recording, so before
the 30 Sep review a bag could be checked while rosbag2 was still stopping.

## 2. Operator guide

Restart the stack after updating to this version: GUI, voice (router + chat),
pipeline, recorder, executor. Until then the old nodes publish no command
events.

**Start**

1. Bring up the stack as usual. Check that the Astra is live; if it is frozen,
   run `ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh'`.
2. **Framing check** (once per session, and after anyone moves a camera):
   Experiments ▸ **🎯 framing check…**. Move the arm **yourself** to home,
   pickup, left, right and hand-over, and press the matching view button at
   each. Both cameras' pictures are saved to `camera_check/` (or
   `vla_experiments/camera_checks/<date>/` outside a trial). Look at them: the
   gripper, the object and the receiving hand must be visible at every place.
   A fresh stream does not prove the framing is useful. This check never
   moves the arm.
3. **⏺ Start trial.** The trial is ready only when the bag has subscribed to
   its topics **and** the recorder has acknowledged **this run's `run_id`**.
   If the recorder refuses (for example a stale camera), the reason appears in
   the window and nothing is recorded.

**During the trial**, mark interventions as they happen:

| button | when |
|---|---|
| ✋ assist (toggle) | you help by hand, e.g. putting the object into the gripper. Press it again when you stop. The interval becomes segment 5 and is never exported as autonomous success. |
| 🔄 rearranged | you moved an object by hand |
| 📷 camera blocked | something happened out of view. Relabelling cannot recover an action nobody saw. |
| ⛔ abort | you aborted what the robot was doing |
| 🤝 hand-over | the robot is handing the object to your hand on purpose, so the hand is part of a valid demonstration |

**Stop**

- Fill in *Full Task Success?* before **⏹ Stop & save**. Without it the
  outcome is decided by the executor's verdict, otherwise it is `unknown`.
- An open assist interval is closed automatically at stop.
- Pressing Stop again, or starting a new trial while the previous one is still
  saving, is refused. It never overwrites.

**Verify the saved trial.** The label under the recorder shows:

- **✔ … complete**: the post-processing status is ok, the bag is verified and
  the manifest is complete.
- **⚠ … NOT verified**, with the reason: the trial is not usable as it is.

Only `postprocess_status.json` decides which of the two is shown. To check a
trial by hand:

```bash
vla_validate ~/morty_ws/experiments/experiments_DD_MM_YYYY/trial_NNN_*   # read-only report
```

## 3. Existing experiments: read-only validation and separate-output repair

The originals are never modified. Validation only reads them. A repair writes
copies with the images symlinked.

```bash
# everything, as a report (nothing is written except the report file)
vla_validate ~/morty_ws/experiments --json /tmp/validation.json

# repaired copies of legacy episodes, OUTSIDE the experiments tree
vla_validate ~/morty_ws/experiments --repair ~/morty_ws/vla_experiments/repaired

# v3 episodes recorded 30 Sep 2026 carry the stricter verdicts of that day's
# recorder (every hand-over "needs review", one late frame blocks the episode).
# Their relabelled copies get today's rules; export from the copies. Name only
# the trials you keep (the workbook's): the exporters read every episode folder
# under the root, including retakes left on disk.
python3 -m vla_experiments.relabel --out ~/morty_ws/vla_experiments/relabelled_30_09 \
    ~/morty_ws/experiments/experiments_30_09_2026/trial_{143,144,146,150}_*

# the older relabeller: dry run by default; --out writes copies; --apply is refused
python3 -m vla_experiments.relabel <episode dir>
python3 -m vla_experiments.relabel --out <dir> <episode dir>
```

`vla_validate` rebuilds each legacy episode's command history from two
sources:

- the router lines in the trial's `logs/voice_gui.log`;
- the executor statuses and phase marks recorded in the episode.

Each instruction is then classified: **confident**, **review** or
**unrecoverable**, with the reason. A repaired copy has
`label_source: reconstructed_from_router_log`, `review_required: true` and
`relabelled_from` (the original path). The exporters therefore skip it
unless `--allow-review` / `MORTY_VLA_ALLOW_REVIEW=1` is set **and** someone
has looked at it.

Results for the collection on 2026-09-30:

- 128 trials checked;
- 37 legacy episodes, with 127 instructions reconstructed, mostly confident;
- 1 instruction unrecoverable;
- 1 broken bag: trial_074, which has no `metadata.yaml`.

In git the bags are LFS pointers, so run the check on a machine that holds the
real files.

## 4. Export options

| lerobot flag | RLDS env | effect |
|---|---|---|
| (default) | (default) | outcome `success`; integrity + alignment ok; labels verified; task segments only |
| `--include-failed/--include-unknown/--include-interrupted` | `MORTY_VLA_OUTCOMES=success,unknown,…` | which outcomes |
| `--allow-quality-issues` | `MORTY_VLA_ALLOW_QUALITY=1` | keep recorder-flagged integrity/alignment issues |
| `--allow-review` | `MORTY_VLA_ALLOW_REVIEW=1` | keep unverified (legacy / reconstructed / flagged) labels |
| `--include-segments hold,reset,…` | `MORTY_VLA_INCLUDE=hold,reset,…` | extra segments exported **with** their instruction |
| `--keep-idle` | — | legacy: the whole episode under its first instruction (wrong labels, only to reproduce old exports) |
| — | `MORTY_VLA_ALL=1` | legacy alias: every outcome + quality + review, **segmentation kept** (it no longer flattens episodes) |

The segmentation by instruction is kept under every option except the explicit
`--keep-idle`.

## 5. Supervised pilot before the next collection

Run the pilot with an operator at the pendant, the e-stop in reach and normal
speed. It is 4 trials with **Record: both**. After each trial:

- the window must say **✔ complete**;
- run `vla_validate <trial>`;
- open `vla_view <trial>` and check that the instruction bar matches what the
  arm did.

1. **Queued moves.** Say "look right and then go home" in one sentence.
   Expect two commands: `look_right` accepted → completed, then `go_home`
   accepted only after that. Each gets its own steps, and `go_home` must not
   cover the look (the trial 114 case).
2. **Similar-object selection.** Use two identical objects. Say "give me the
   vitamin tube", answer the which-one question by pointing, then say "yes".
   Expect the first command `rejected` with the question as its reason, a
   `clarify` segment, and the yes as the accepted pick with
   `target.proposal_id` set. Say "and then give it to me and go home" during
   the approach: it must be `deferred` and then accepted at the release pause
   as a child of the pick (the trial 121 case).
3. **Hand-over with confirmation and return home.** Say "bring it home and
   hand it over", press 🤝 when the robot offers the object, and say "yes" at
   "Should I open the gripper?". Expect one instruction covering the carry,
   hold and release, with the hand visible in the Astra view.
4. **Interruption.** Start a pick and say "stop" during the approach, then
   press Stop & save without a verdict. Expect the command `cancelled` /
   `interrupted`, the trial outcome `unknown` or `interrupted`, and a reset
   span that ends where the arm stops.

Any mismatch means stop the collection and look at `meta["commands"]` and
the bag's `/robot/command_event` before recording more.

## 6. Limitations

- **Hardware.** No hardware test has been run for this version. The router,
  recorder, labeller and GUI logic are tested offline and on an isolated ROS
  domain with a fake executor and pipeline. The pilot above is the first real
  check.
- **Old experiments.** Trials recorded before 2026-09-30 have no command
  events. Their labels can only be reconstructed from logs, so they stay
  `review_required`.
- **Executor phases.** The executor (C++) is unchanged. Its task-state phases
  come from the existing latched topic, and the detailed phase marks still
  arrive only at the end of a cycle.
- **Named-pose moves.** Their motion interval is inferred from joint
  motion between accepted and completed.
- **Success means the operator's verdict.** An accepted goal or a closed
  gripper does not prove that the right object was taken; only the operator's
  verdict does.
- **Out-of-view actions.** An action outside both camera views cannot be
  repaired afterwards. Mark it with 📷 and exclude the trial.
