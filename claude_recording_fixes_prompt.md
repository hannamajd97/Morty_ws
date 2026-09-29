Implement and validate the recording fixes below in this Morty_ws ROS 2 / UR3e repository. The purpose is to collect reliable demonstrations for future OpenVLA fine-tuning. Treat this as an implementation task: inspect the current code, make the necessary changes, and test them.

1. Scope and working rules

Read AGENTS.md and relevant repository documentation first. Inspect git status and preserve existing uncommitted work. The findings below came from a review on 29 September 2026; verify them against the current code because development is ongoing.

Preserve original experiments and recordings. Make legacy-data analysis read-only by default and write repaired datasets to a separate output directory. Keep the current manipulation behavior, motion planning, interlocks and emergency-stop behavior intact while adding recording instrumentation. Use offline tests, mocks and replay for verification; physical robot tests require an operator's explicit authorization and supervision.

Cover GUI recording, standalone/manual recording and automatic recording modes. Command transcripts are the language source; fixes must work without audio files. Record every command, including commands issued during another task.

Follow the repository's readability preferences. Keep new scripts procedural, with configuration, loading, processing and output visible in order. Use functions for substantial shared or independently testable operations; avoid unnecessary layers or unrelated refactoring.

2. Inspect the complete recording path

Start with these files and follow their actual dependencies:

- src/vla_experiments/vla_experiments/episode_recorder.py
- src/vla_experiments/vla_experiments/raw_dataset.py
- src/vla_experiments/vla_experiments/relabel.py
- src/vla_experiments/vla_experiments/postprocess.py
- src/vla_experiments/vla_experiments/joints_full.py
- src/vla_experiments/config/recorder.yaml
- src/vla_experiments/rlds_builder/morty_ur3e/morty_ur3e_dataset_builder.py
- src/morty_gui/morty_gui/mmhri_window.py
- src/morty_gui/morty_gui/vla_link.py
- src/morty_gui/morty_gui/bag_link.py
- src/morty_gui/morty_gui/managed_launch.py
- src/morty_gui/morty_gui/config.py
- src/morty_gui/morty_gui/trial_postprocess.py
- src/morty_gui/morty_gui/trial_times.py
- src/communication_llm/communication_llm/chat_node.py
- src/intent_router/intent_router/intent_router_node.py
- src/grasp_pipeline/grasp_pipeline/grasp_selection_node.py
- src/grasp_pipeline/grasp_pipeline/grasping/pick_pipeline.py
- src/ur3e_gripper_3d_moveit_commander/src/grasp_executor_lin_llm.cpp

Also update affected readers, viewers and converters when changing the data contract.

Verified examples from experiments/experiments_29_09_2026, if available:

- Trial 114: look_right and go_home intents were published about 8 ms apart. Current labeling assigns zero frames to look_right even though that movement occurred.
- Trial 121: motion_start, close_sent and lift_done at approximately 37.02, 39.13 and 41.74 seconds receive the instruction “open the gripper and go back home.” This is a recording-label defect; do not assume the current speech parser still reproduces the historical mistaken intent.
- Trial 126: open_gripper with then=release_home becomes only “open the gripper,” including the return-home movement.
- Trials 116 and 117 have bag metadata but no database. Trial 124 has a malformed database. The current readiness function nevertheless reports their bags ready. These are dialogue-only trials; absence of a VLA episode itself is not their defect.
- All 13 VLA episodes have quality.ok=true despite the labeling problems.

These examples establish failures to handle; they do not establish the cause of the historical bag corruption or guarantee that every old label can be reconstructed automatically.

3. Make instruction intervals follow actual execution

The recorder currently activates instructions when intents arrive. The router can queue, deduplicate, reject or defer those same intents. Replace this assumption with explicit command lifecycle recording.

Define and document a versioned event contract. Include unique run and command IDs; transcript/utterance and question IDs where applicable; cycle and parent/child plan relationships; event identity/sequence; source timestamp and clock domain; receipt timestamp; action, attributes and resolved target; lifecycle state, outcome and reason.

Record received, accepted/queued, execution-started, completed, rejected, cancelled and superseded events at the component that knows each fact. Instrument standalone named-pose moves and gripper commands as well as grasp cycles. A recorded goal acceptance must not be mislabeled as proof that motion already started.

Associate the router, selection pipeline, executor, recorder and bag records unambiguously. Handle follow-up commands, repeated commands, interruptions, node restarts, duplicated events and delayed/out-of-order delivery. Preserve raw intents and transcripts even when they are rejected.

Derive training instruction intervals from the accepted command actually executing. A queued or deferred command must not replace the current action's label. Connect “yes” to the question and task it answers. Use explicit associations instead of nearest-text or last-message guessing for new recordings.

Document observation/action boundary conventions and preserve causality around command changes. When execution evidence is insufficient, flag the segment for review instead of inventing an interval.

4. Preserve resolved targets and complete compound commands

Save the final accepted object selection, including available distinguishing attributes, selection source, proposal ID, selected image timestamp, bounding box and image coordinate frame. Preserve object-instance identity where the existing system provides it; an index is meaningful only with its associated frame/detection set.

Enrich an existing generic spoken pick when pointing or clarification resolves it. Do not return early and retain an ambiguous class-only instruction. Generate a distinguishing instruction only from supported evidence; preserve the original transcript and resolution provenance. An unresolved target must remain explicitly unresolved.

Normalize the complete accepted command/plan consistently across execution, recording, relabeling and export. Cover all supported action/attribute combinations, including open_gripper + then=release_home, go_home + then=release, destination answers and pick/place follow-ups. Prevent the then attribute from disappearing from labels.

Choose and document whether a compound instruction stays intact across its executed sequence or becomes separate correctly timed subcommands. Keep the parent plan and execution relationship either way. Do not teach a desired action as successful when the robot actually executed a conflicting action.

5. Record phases incrementally and segment them correctly

Consume the existing /grasp_executor_lin_llm/task_state information and record detailed phase transitions as they happen. Currently detailed timing marks are emitted at cycle end, which is insufficient for interrupted recordings.

Distinguish setup, clarification, waiting for destination, waiting for release confirmation, task motion, gripper actions, intended handover, user-requested return home, automatic cleanup/reset, and intervention.

Preserve all raw frames. Exclude unrelated clarification/setup and automatic cleanup from default manipulation training. Keep meaningful task holds and handover waiting; stationary frames alone are not an error. A requested return-home action remains part of its task. An automatic post-handover close must not remain labeled “open the gripper.”

Handle completion/timing events arriving in either order across ROS topics. Finalization must reconcile the relevant cycle's events or report incomplete evidence after a bounded timeout. Do not silently ignore late completion evidence and declare the episode valid.

6. Align camera observations, robot state and actions

The recorder currently samples independent latest values and checks camera age mainly from receipt time. Preserve source and receipt timestamps and explicitly handle their clock domains. Validate clock comparability across the camera relay and robot computers.

Implement a documented alignment policy using bounded timestamped buffers or synchronized resampling. Associate images and robot state with the intended observation time, record alignment error, and enforce configurable tolerances. Preserve the raw evidence needed to inspect alignment.

Measure capture delay, inter-camera skew, stale states, source timestamp discontinuities and repeated source frames. Identical image content alone does not prove a stale camera. Track gripper-state freshness independently from arm-joint freshness.

Keep acquisition responsive and bounded in memory/CPU usage. Avoid restoring excessive high-rate Python callback load. Document coordinate frames, units, gripper convention and action interval. Exported action deltas must remain consistent with the selected sample interval; do not derive actions across excluded gaps or unrelated commands.

7. Make recording startup, stopping and completion dependable

Implement explicit preparing, recording, saving, complete and incomplete/error states associated with a unique run ID.

Announce readiness only after the relevant recorders acknowledge the current run and required streams are available. A cached status, old log message or elapsed timeout is not an acknowledgment. Separate required continuous sensor streams from optional/event-only topics. Determine expected artifacts from the run configuration and acknowledgments, not merely from which folders happen to exist during polling.

On Stop, preserve the operator's requested recording boundary. If a command remains active, record the interruption and appropriate unknown/incomplete outcome. Do not silently equate stopping with task success. Keep the GUI responsive while waiting for finalization.

Wait for the bag process to finish and inspect its exit status. Verify every database referenced by its metadata, database readability/integrity, required topic coverage and plausible recording timestamps. metadata.yaml alone is insufficient.

Verify the expected images, image decodability, array lengths, metadata and required finite values. Respect optional streams and dialogue-only trials. Report recording integrity separately from physical task outcome and training eligibility. Propagate postprocessing errors to the GUI instead of returning success after logging an exception.

8. Fix write completion, crash recovery and trial reuse

_ImageWriter.drain currently returns when its queue becomes empty, although the worker may still be writing the last item. Wait for all outstanding writes, including in-progress work, with explicit timeout/failure reporting. Check cv2.imwrite's Boolean return in the raw-image fallback as well as exceptions in both image paths.

Write arrays and metadata through temporary files and atomic replacement. Publish a completion manifest/marker only after the corresponding consistent set of files is finalized and validated. Apply equivalent protection to joints_full, relabeling and other postprocessing rewrites. Preserve recoverable incomplete recordings and report them accurately after interruption or restart.

Prevent a repeated trial ID or start request from deleting a completed episode. Refuse accidental reuse or allocate a separate unique run directory, preserving the matching bag, images and metadata. Handle repeated Stop requests and overlapping save/start requests without overwriting another run.

Record schema, recording configuration, software revision/dirty status and relevant camera calibration/configuration provenance with the run.

9. Preserve missing evidence in the ROS bag

Add /gripper_controller/command_event and /grasp_executor_lin_llm/task_state to the bag topic configuration. Also record the new command lifecycle and resolved-selection events.

/speech/utterance_meta and /grasp_pipeline/point_proposal are already recorded in the reviewed configuration. Preserve them and connect their identities/context to VLA metadata. Verify publisher/subscriber QoS compatibility and behavior when optional topics are absent.

10. Strengthen quality, outcomes and export behavior

Separate file integrity, temporal alignment, semantic review and task outcome. quality.ok must not imply verified correct instruction/action matching merely because arrays are readable.

Validate command/execution correspondence, resolved targets, segment continuity, completion evidence, image/state/action validity and outcome provenance. Flag contradictions and uncertain legacy evidence. Distinguish checks software can prove from cases requiring operator review. Preserve intended target/destination separately from observed execution and flag disagreements.

Use explicit outcomes such as success, failure, unknown, interrupted or the repository's documented equivalent. A bare or misspelled Stop argument must not default to human-confirmed success. Record per-command/segment outcomes as well as the trial outcome. Goal-server success or gripper closure alone does not establish correct-object/task success.

Fix MORTY_VLA_ALL: it currently includes failed/unknown/flagged episodes and passes keep_idle=True, which puts the entire episode under its first instruction. Separate outcome overrides, quality overrides, idle inclusion and reset inclusion. Preserve valid instruction segmentation under every supported override. Handle the old option explicitly and document its behavior.

Keep exported segments and their actions temporally coherent. Update relabeling to use the corrected shared rules; merely rerunning the old _label_steps function is not a repair.

11. Support camera checks and human-intervention annotations

Add a simple recorded operator check of both camera views through the actual home, pickup, left, right and handover workspace. Stream freshness must not be presented as proof of useful framing. The check must not initiate robot movement automatically.

Provide timestamped annotations, including start/end intervals where appropriate, for manual assistance, object rearrangement, camera visibility problems, aborts and intended handovers.

Keep intended receiving hands in valid handover demonstrations. Mark manually placing an object into the gripper as assistance and exclude the affected interval from autonomous-grasp success training. Preserve the raw record. Document that an out-of-view action cannot be recovered by relabeling.

12. Preserve compatibility and make legacy repair reviewable

Version the changed schema and update affected consumers. Keep old recordings readable without silently claiming they contain the new execution evidence.

Provide a dry-run validation/repair report for existing episodes. Any repair operation must preserve originals and write to a separate output location. Record the evidence and provenance of each proposed correction. Separate confidently repairable segments, segments needing review and unrecoverable missing evidence.

Do not promise automatic recovery of trials 114/121 merely by adding future command IDs. Use existing authoritative evidence where available; otherwise report uncertainty. Do not bulk-relabel the historical collection as part of this implementation.

13. Test the required behavior

Add focused regression tests and meaningful mocked/replay integration checks, following the repository's testing conventions. Cover:

- Two commands published milliseconds apart but executed sequentially: each actual movement receives its own correct instruction.
- Queued, rejected, duplicate, deferred and cancelled commands: none incorrectly takes over another action's label.
- A pending release while a grasp is executing: the grasp/lift retains the pick instruction.
- Compound commands: opening plus return home and home plus handover retain all intended executed parts.
- Pointing/clarification between similar objects and yes/no answers: target and question associations survive recording/export.
- Clarification waits, meaningful handover holds, requested home motion, automatic cleanup and manual assistance receive the correct segment treatment.
- Phase/completion events arriving out of order or near Stop; interruptions and restarts preserve evidence and honest outcomes.
- Freshly received images with old capture stamps, stale gripper state, clock/skew violations and repeated source frames are detected.
- The image queue is empty while a write remains in progress: completion must wait. False imwrite returns and write exceptions produce explicit failures.
- Bag metadata exists but its database is missing/corrupt; bag startup fails; postprocessing fails: the UI never reports verified completion.
- Reused trial IDs, repeated Stop and a new Start during saving preserve existing data and run associations.
- Every export override preserves instruction segmentation and action boundaries; interrupted/unreviewed segments are handled according to explicit eligibility rules.
- Legacy readers remain usable; repair defaults to dry-run and leaves original files unchanged.

Use the named historical trials read-only where available and deterministic fixtures otherwise. Do not fabricate historical execution events to make tests pass. Test the implementation's real code paths.

Run relevant Python checks, package tests and builds for modified ROS/C++ components. Clearly distinguish passed offline tests, ROS integration tests, and hardware tests still requiring an operator. Provide a short supervised pilot protocol covering queued moves, similar-object selection, handover with confirmation/return home, and interruption before the next larger collection.

14. Deliver the completed work clearly

Before editing, briefly describe the current end-to-end recording flow and your implementation sequence. Then proceed with the work; do not stop at a proposed plan. Resolve routine choices from the repository and these requirements, and identify genuine blockers explicitly.

At completion provide:
- An issue-by-issue table: fixed, already correct, or unresolved, with evidence.
- Changed files and a plain-language explanation of the resulting behavior.
- The event/schema contract and configuration changes.
- Test/build results, including failures or checks not run.
- A concise operator guide for starting, stopping, checking framing, annotating interventions and verifying a saved trial.
- Instructions for read-only validation and separate-output repair of existing experiments.
- Remaining limitations and the supervised checks required before collecting more training data.

Complete all applicable requirements. Never substitute a green recording-health flag for evidence that a demonstration is semantically correct.
