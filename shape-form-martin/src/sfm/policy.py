"""sfm.policy - tracked policy of where campaign code may run unguarded (review finding A5).

LAPTOP_MACHINE_IDS lists the machine ids (``sfm.machine.machine_tag().id``: GPU, SM count, launch grid, torch / CUDA /
cuDNN / Python build, TF32 flags - no host names) of the laptop of record. On every OTHER machine, campaign code
refuses to start unless

    * the output root lies outside the clone (``paths.pristine_mode()``: ``SFM_OUT_ROOT`` set to a directory outside
      the repository), and
    * the file-access guard is active (``sfm.guard.active()``: ``SFM_GUARD_AREA`` set),

both of which ``scripts/launch.sh`` provides. There is no opt-out: a command started without the launcher on the H200
stops before it can write a second registry or reference set inside the clone, or run unguarded. Only processes
that run on the CPU through the explicit unit-test opt-in (``SFM_ALLOW_CPU=1`` in ``sfm.train.fit``) are exempt;
they cannot produce a number of record (their tag says ``gpu_name: "cpu"``).

If the laptop's software changes (torch, Python, ...), its id changes and it is treated like any other machine until
this file is updated (fail closed).
"""
LAPTOP_MACHINE_IDS = ("rtx-4060-laptop-gpu-g144-76c0f59a",)
