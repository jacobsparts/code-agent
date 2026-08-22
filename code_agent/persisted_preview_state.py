import copy
from dataclasses import dataclass


@dataclass
class PersistedPreviewState:
    definitions: dict[int, tuple[str, str]]
    active_placements: dict[tuple[int, int], int]
    exec_start_seq: int = 0

    @classmethod
    def empty(cls, exec_start_seq: int = 0):
        return cls({}, {}, exec_start_seq)

    def placement_status(
        self,
        preview_event_seq: int,
        source_start_seq: int,
        source_end_seq: int,
    ) -> str:
        requested = (source_start_seq, source_end_seq)
        existing = self.active_placements.get(requested)
        if existing is not None:
            return "duplicate" if existing == preview_event_seq else "apply"
        for start, end in self.active_placements:
            if source_start_seq <= end and start <= source_end_seq:
                contains = source_start_seq <= start and end <= source_end_seq
                contained = start <= source_start_seq and source_end_seq <= end
                if not contains and not contained:
                    return "overlap"
                if contained:
                    return "inside"
        return "apply"

    def install_placement(
        self,
        preview_event_seq: int,
        source_start_seq: int,
        source_end_seq: int,
    ):
        self.active_placements[(source_start_seq, source_end_seq)] = preview_event_seq


class PersistedPreviewTransitions:
    def __init__(self):
        self.state = PersistedPreviewState.empty()
        self.snapshots = {0: copy.deepcopy(self.state)}

    def apply(
        self,
        *,
        seq,
        event_type,
        payload,
        has_preview_blob,
        apply_placement=None,
    ):
        placement = None
        if type(seq) is not int:
            return None
        if not isinstance(payload, dict):
            self.snapshots[seq] = copy.deepcopy(self.state)
            return None

        if event_type == "preview_created":
            if set(payload) == {"preview_key", "summary"}:
                key = payload.get("preview_key")
                summary = payload.get("summary")
                if (
                    isinstance(key, str)
                    and key
                    and isinstance(summary, str)
                    and summary.strip()
                    and has_preview_blob(key)
                ):
                    self.state.definitions[seq] = (key, summary)

        elif event_type == "preview_placed":
            if set(payload) == {
                "preview_event_seq",
                "source_start_seq",
                "source_end_seq",
            }:
                preview_event_seq = payload.get("preview_event_seq")
                source_start_seq = payload.get("source_start_seq")
                source_end_seq = payload.get("source_end_seq")
                definition = self.state.definitions.get(preview_event_seq)
                if (
                    definition is not None
                    and type(preview_event_seq) is int
                    and type(source_start_seq) is int
                    and type(source_end_seq) is int
                    and preview_event_seq < seq
                    and self.state.exec_start_seq
                    < source_start_seq <= source_end_seq < preview_event_seq
                ):
                    status = self.state.placement_status(
                        preview_event_seq,
                        source_start_seq,
                        source_end_seq,
                    )
                    if status == "duplicate":
                        placement = ("duplicate", preview_event_seq, definition, source_start_seq, source_end_seq)
                    elif status == "apply":
                        accepted = True if apply_placement is None else bool(
                            apply_placement(
                                preview_event_seq,
                                definition,
                                source_start_seq,
                                source_end_seq,
                            )
                        )
                        if accepted:
                            self.state.install_placement(
                                preview_event_seq,
                                source_start_seq,
                                source_end_seq,
                            )
                            placement = ("apply", preview_event_seq, definition, source_start_seq, source_end_seq)

        elif event_type == "rewind":
            target_seq = payload.get("target_seq")
            if type(target_seq) is int:
                self.state = copy.deepcopy(
                    self.snapshots.get(target_seq, self.snapshots[0])
                )

        elif event_type == "exec":
            self.state = PersistedPreviewState.empty(seq)

        self.snapshots[seq] = copy.deepcopy(self.state)
        return placement
