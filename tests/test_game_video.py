"""The desk video a game can record (mahjong/desk_video.py, --record-desk) and the game's video made from a session
(scripts/game_video.py): end to end on simulated games, and the parts that keep the video honest."""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mahjong import config, desk_video, privacy
from mahjong.perception import RealSense
from mahjong.session import read_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import game_video  # noqa: E402


def frames_in(path):
    capture = cv2.VideoCapture(str(path))
    n = 0
    while capture.read()[0]:
        n += 1
    return n


@pytest.fixture
def fast_desk(monkeypatch):
    monkeypatch.setattr(desk_video, "FPS", 20.0)       # a sim hand takes ~2s


# --- the recorder ---

def test_the_recorder_keeps_frames_and_their_times_and_prefers_a_scans_frame(tmp_path):
    grabbed = []

    def grab():
        grabbed.append(1)
        return np.full((48, 64, 3), 100, np.uint8)

    recorder = desk_video.DeskRecorder(tmp_path, grab, fps=50).start()
    recorder.offer(np.zeros((48, 64, 3), np.uint8))
    time.sleep(0.3)
    result = recorder.stop()
    times = json.loads((tmp_path / desk_video.TIMES).read_text())["times"]
    assert result["frames"] == len(times) == frames_in(tmp_path / desk_video.VIDEO) > 3
    assert len(grabbed) == len(times) - 1                  # the offered frame went in without a grab of its own
    assert times == sorted(times)


def test_a_failing_camera_never_stops_the_game(tmp_path):
    def grab():
        raise RuntimeError("Frame didn't arrive")

    recorder = desk_video.DeskRecorder(tmp_path, grab, fps=50).start()
    time.sleep(0.2)
    result = recorder.stop()
    assert result["frames"] == 0 and result["failures"] > 0 and "Frame didn't arrive" in result["last_error"]
    assert result["video"] is None


class FakeFrames:
    def __init__(self, image):
        self.image = image

    def get_color_frame(self):
        return self

    def get_data(self):
        return self.image


class FakePipeline:
    def __init__(self):
        self.polls = 0

    def poll_for_frames(self):
        self.polls += 1
        return FakeFrames(np.full((4, 4, 3), 7, np.uint8))


def bare_realsense():
    rs = RealSense.__new__(RealSense)       # no device: only what idle_colour reads
    rs.pipeline, rs._lock, rs._last_pull, rs.on_colour = FakePipeline(), threading.Lock(), 0.0, None
    return rs


def test_the_desk_video_never_takes_a_frame_from_a_scan():
    rs = bare_realsense()
    rs._last_pull = time.monotonic()            # a scan just pulled a frame: its next frames stay consecutive
    assert rs.idle_colour(idle_s=1.0) is None
    with rs._lock:                              # a scan pulling right now
        rs._last_pull = 0.0
        assert rs.idle_colour() is None
    assert rs.pipeline.polls == 0
    frame = rs.idle_colour()
    assert frame is not None and frame.shape == (4, 4, 3) and rs.pipeline.polls == 1


# --- the game's video ---

def play_sim(tmp_path, monkeypatch, argv):
    import play
    monkeypatch.setattr(config, "SESSIONS", tmp_path)
    p = play.Player(play.parse_args(["--sim", *argv]), ask=lambda prompt: "")
    try:
        assert p.play() == "clean"
    finally:
        p.close()
    return p.session.dir


def test_a_sim_game_with_record_desk_keeps_a_desk_video_and_makes_a_game_video(tmp_path, monkeypatch, fast_desk):
    folder = play_sim(tmp_path, monkeypatch, ["--record-desk"])
    events = read_events(folder)
    end = next(e for e in events if e["type"] == "desk_video_end")
    times = json.loads((folder / desk_video.TIMES).read_text())["times"]
    assert end["frames"] == len(times) == frames_in(folder / desk_video.VIDEO) > 0 and end["failures"] == 0
    assert times == sorted(times)
    assert [e["type"] for e in events].index("desk_video_end") < [e["type"] for e in events].index("session_end")

    attempts = [e["attempt_dir"] for e in events if e["type"] == "attempt"]
    for attempt in attempts:                    # the sim's policy saves no gripper video: 10 frames each
        video = cv2.VideoWriter(str(folder / attempt / "wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for _ in range(10):
            video.write(np.full((48, 64, 3), 200, np.uint8))
        video.release()
    result = game_video.build(folder, "story", clip_speed=2)
    assert len(attempts) == 3 and result["desk"] == "desk.mp4" and result["clips"] == 3     # GD, RD and 9B were picked up
    cards = result["card_frames"]
    assert result["frames"] == cards + len(times) + 3 * 5           # every desk frame once at 30 a second; clips at 2x
    assert result["path"] == folder / "video" / "game_story.mp4"
    assert frames_in(result["path"]) == result["frames"] > (game_video.TITLE_S + game_video.END_S) * game_video.OUT_FPS


def test_without_record_desk_the_video_is_made_from_the_scans(tmp_path, monkeypatch):
    folder = play_sim(tmp_path, monkeypatch, [])
    assert not (folder / desk_video.VIDEO).exists()
    scans = game_video.scan_frames(folder, read_events(folder))
    result = game_video.build(folder, "timelapse", fps=10, out=tmp_path / "t.mp4")
    assert result["desk"] == f"{len(scans)} scans" and result["clips"] == 0
    cards = result["card_frames"]
    assert result["frames"] == cards + len(scans) * 3          # each scan held 3 frames at 10 a second
    assert frames_in(tmp_path / "t.mp4") == result["frames"]


def test_face_down_play_records_the_desk_too(tmp_path, monkeypatch, fast_desk):
    from test_play_face_down import HAND, LIGHT, run, script_in
    script = script_in(monkeypatch, tmp_path, HAND + ["4D", "8B"])
    folder, _ = run(script, ["--record-desk"], [*LIGHT, "", "", "", "", "", "end", ""])
    events = read_events(folder)
    end = next(e for e in events if e["type"] == "desk_video_end")
    assert end["frames"] > 0 and (folder / desk_video.VIDEO).exists()
    # and its video counts the hand's pickups and turns, and cuts its gripper clips in (2026-11-19: '0 of 0 pickups', no clips)
    attempts = [e["attempt_dir"] for e in events if e["type"] == "attempt"]
    for attempt in attempts:                    # the sim's policy saves no gripper video
        video = cv2.VideoWriter(str(folder / attempt / "wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for _ in range(4):
            video.write(np.full((48, 64, 3), 200, np.uint8))
        video.release()
    result = game_video.build(folder, "story", clip_speed=2, out=tmp_path / "fd.mp4")
    assert result["clips"] == len(attempts) == 13 + 2 * 2       # the 13 read, then each turn's draw and its pick off the zone
    story = game_video.Story(folder, events)
    assert story.counts()["picked"] == story.counts()["attempts"] == len(attempts) and story.counts()["turns"] == 2
    assert story.hand_lines(game_video.seconds(events[-1]["time"]))[-1].startswith("Discards: ")


def event(time, type, **fields):
    return {"time": f"2026-11-04T18:{time}", "type": type, **fields}


EVENTS = [
    event("00:00", "session_start", policy="default"),
    event("00:01", "bench_setup", policy="v6"),
    event("00:02", "intervention", kind="name_asked", attempt_dir=None),
    event("01:00", "discard_chosen", turn=1, drawn="1B", tile_name="7B"),
    event("01:30", "attempt", tile="7B", success=True, attempt_dir="001_180100_7B"),
    event("01:40", "intervention", kind="straightened", attempt_dir="001_180100_7B"),
    event("01:50", "turn_done", turn=1, discarded="7B"),
    event("02:00", "attempt", tile="9C", success=False, attempt_dir="002_180150_9C"),
    event("03:00", "hand_end", result="win", turns=1, grade="rescued"),
    event("03:01", "session_end", reason="quit"),
]


def test_the_captions_say_what_the_robot_was_doing_and_count_what_a_person_did(tmp_path):
    story = game_video.Story(tmp_path / "20261104_180000_play", EVENTS)
    t = game_video.seconds
    assert story.doing(t("2026-11-04T18:00:30")) == "Reading the hand"
    assert story.doing(t("2026-11-04T18:01:10")) == "Turn 1: drew 1B, discarding 7B"
    assert story.doing(t("2026-11-04T18:01:55")) == "Turn 1: discarded 7B"
    assert story.counts(t("2026-11-04T18:01:35")) == {"picked": 1, "attempts": 1, "turns": 1, "questions": 1, "rescues": 0,
                                                     "resets": 0, "hands_on": 0}
    assert story.counts() == {"picked": 1, "attempts": 2, "turns": 1, "questions": 1, "rescues": 1, "resets": 0, "hands_on": 1}
    ending = story.ending()
    assert ending[0] == "Result: win in 1 turns" and "Graded: rescued" in ending
    assert "1 of 2 pickups held the tile" in ending[2]
    assert "stepped in by hand 1 time (1 rescue, 0 resets of the mat)" in ending[4]
    assert ending[-1] == "session 20261104_180000_play"
    assert "policy v6" in story.title()


FACE_DOWN_EVENTS = [
    event("00:00", "session_start", kind="play_fd", policy="v13", unattended=False),
    event("00:30", "crude_pick", turn=0, step="register", spot="S1", held=True, attempt_dir="001_180030_reg_S1"),
    event("00:50", "fd_hand_confirmed", hand={"S1": "1C", "S2": "2C"}, fixed={}, by="person"),
    event("01:00", "crude_pick", turn=1, step="draw", spot="F", held=False, attempt_dir="002_180100_t1_draw_top"),
    event("01:20", "crude_pick", turn=1, step="draw", spot="F", held=True, attempt_dir="003_180120_t1_draw_top"),
    event("01:30", "crude_read", turn=1, guess="4D", name="4D", face_hidden=False, guess_confidence=0.9),
    event("01:40", "crude_choice", turn=1, drawn="4D", discard="4D", slot=None, shanten_after=2),
    event("03:00", "hand_end", result="end", turns=1, grade="clean", hand={"S1": "1C", "S2": "2C"}, discards=["4D"]),
    event("03:01", "session_end", reason="done"),
]


def test_a_face_down_hand_s_picks_turns_and_clips_are_in_its_video(tmp_path):
    """play_face_down.py logs crude_pick and crude_choice, not play.py's attempt and discard_chosen (post-merge review of
    #127: a 135-minute face-down hand's video said 'Reading the hand' throughout, '0 of 0 pickups', no gripper clips)."""
    folder = tmp_path / "20261104_180000_play_fd"
    for e in FACE_DOWN_EVENTS:
        if e.get("attempt_dir"):
            (folder / e["attempt_dir"]).mkdir(parents=True)
            (folder / e["attempt_dir"] / "wrist.mp4").write_bytes(b"")
    story = game_video.Story(folder, FACE_DOWN_EVENTS)
    t = game_video.seconds
    assert story.doing(t("2026-11-04T18:00:40")) == "Reading the hand"
    assert story.doing(t("2026-11-04T18:02:00")) == "Turn 1: drew 4D (checked), discarding 4D"      # a person confirmed it
    assert story.hand_lines(t("2026-11-04T18:02:00")) == ["Hand: 1C 2C"]
    assert story.counts()["picked"] == 2 and story.counts()["attempts"] == 3 and story.counts()["turns"] == 1
    assert "A person checked 3 of the reader's names" in story.ending()
    shown = game_video.clips(folder, FACE_DOWN_EVENTS)
    assert [(c["step"], c["spot"], c["success"]) for c in shown] == [("register", "S1", True), ("draw", "F", False), ("draw", "F", True)]
    assert game_video.clip_label(shown[0], 4) == "gripper camera, 4x speed: reading S1, held"
    assert game_video.clip_label(shown[1], 4) == "gripper camera, 4x speed: drawing from the wall, missed"


def shot(time, number, tile, spot, turn, guess=None, confidence=0.9, by="reader"):
    files = [f"photos/{tile}/unknown_h{number}_{k}.jpg" for k in range(3)]
    return event(time, "holdup_shot", tile=tile, kind="unknown", hold=number, files=files, shot=number, spot=spot, turn=turn,
                 frame_poses=["base_view_1", "base_view_2", "base_view_9"], review_first=True, filed_by=by,
                 guess=guess or tile, guess_confidence=confidence)


READ_EVENTS = [
    event("00:00", "session_start", kind="play_fd", policy="v13", unattended=True),
    event("00:30", "crude_pick", turn=0, step="register", spot="S1", held=True, attempt_dir="001_180030_reg_S1"),
    event("00:40", "crude_pick", turn=0, step="register", spot="S2", held=True, attempt_dir="002_180040_reg_S2"),
    shot("00:50", 1, "1C", "S1", 0), shot("00:50", 2, "2C", "S2", 0),
    event("00:50", "fd_hand_confirmed", hand={"S1": "1C", "S2": "2C"}, fixed={}, by="reader"),
    event("00:50", "fd_turn_start", turn=1, slots=["S1", "S2"], hand={"S1": "1C", "S2": "2C"}, discards=[]),
    event("01:00", "crude_pick", turn=1, step="draw", spot="F", held=True, attempt_dir="003_180100_t1_draw_top"),
    shot("01:30", 3, "3C", "draw t1", 1),
    event("01:30", "crude_read", turn=1, guess="3C", name="3C", face_hidden=False, guess_confidence=0.9, by="reader"),
    event("01:30", "crude_choice", turn=1, drawn="3C", discard="1C", slot="S1", shanten_after=1),
    event("01:30", "crude_pick", turn=1, step="pick_from_hand", spot="S1", held=True, attempt_dir="004_180130_t1_S1"),
    event("02:00", "fd_turn_start", turn=2, slots=["S1", "S2"], hand={"S1": "3C", "S2": "2C"}, discards=["1C"]),
    event("02:00", "crude_pick", turn=2, step="draw", spot="F", held=True, attempt_dir="005_180200_t2_draw_top"),
    shot("02:30", 4, "4C", "draw t2", 2, guess="5C", confidence=0.41, by="person"),
    event("02:30", "crude_win", turn=2, drawn="4C"),
    event("03:00", "hand_end", result="win", turns=2, grade="clean", hand={"S1": "3C", "S2": "2C"}, discards=["1C"]),
    event("03:01", "session_end", reason="quit"),
]
BOXES = {"base_view_1": [10, 0, 50, 40], "base_view_2": [20, 0, 60, 40]}       # base_view_9 has none: its photo stays out


def read_session(tmp_path):
    folder = tmp_path / "20261104_180000_play_fd"
    for e in READ_EVENTS:
        if e.get("attempt_dir"):
            (folder / e["attempt_dir"]).mkdir(parents=True)
            video = cv2.VideoWriter(str(folder / e["attempt_dir"] / "wrist.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            for _ in range(4):
                video.write(np.full((48, 64, 3), 200, np.uint8))
            video.release()
        for k, f in enumerate(e.get("files", [])):
            (folder / f).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f), np.full((48, 64, 3), 40 * k, np.uint8))
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in READ_EVENTS))
    return folder


def test_each_draw_s_base_camera_photos_come_after_its_gripper_clip_and_before_the_next(tmp_path):
    """The owner's ask (2026-11-19): a viewer sees what the base camera saw of a drawn tile, not only the name the robot took."""
    folder = read_session(tmp_path)
    reads = game_video.base_reads(folder, READ_EVENTS, boxes=BOXES)
    assert [r["where"] for r in reads] == ["S1", "S2", "draw t1", "draw t2"]
    assert all(len(r["crops"]) == 2 for r in reads)                # one per pose with a tile box: the room stays out
    drawn = [r for r in reads if game_video.is_draw(r)]
    order = game_video.cut_ins(game_video.clips(folder, READ_EVENTS), drawn)
    assert [c.get("step") or c["where"] for c in order] == [
        "register", "register", "draw", "draw t1", "pick_from_hand", "draw", "draw t2"]      # the photos first on a tie
    assert game_video.read_label(drawn[0]) == "base camera: the reader read 3C (90% sure)"
    assert game_video.read_label(drawn[1]) == "base camera: the reader read 5C (41% sure), a person named it 4C"
    assert game_video.read_label({**drawn[0], "truth": "8C"}).endswith("; reviewed since: 8C")


def test_the_hand_s_photos_follow_each_slot_through_the_turns(tmp_path):
    folder = read_session(tmp_path)
    reads = game_video.base_reads(folder, READ_EVENTS, boxes=BOXES)
    hand = game_video.final_hand(READ_EVENTS, reads)
    assert [(t["name"], t["read"]["where"], t["drawn"]) for t in hand] == [
        ("3C", "draw t1", False), ("2C", "S2", False), ("4C", "draw t2", True)]     # S1's 1C was thrown on turn 1


def test_a_winning_hand_s_photos_are_grouped_by_meld_with_the_draw_in_its_meld():
    names = ["1C", "2C", "3C", "4D", "5D", "6D", "7B", "8B", "9B", "EW", "EW", "EW", "RD", "RD"]
    tiles = [{"name": n, "read": None, "drawn": False} for n in names[:-1]] + [{"name": "6D", "read": None, "drawn": True}]
    tiles[5]["name"] = "RD"         # the 13 held RD RD before the draw; 6D completes 4D 5D 6D
    groups = game_video.hand_groups(tiles, won=True)
    assert [[t["name"] for t in g] for g in groups] == [
        ["1C", "2C", "3C"], ["4D", "5D", "6D"], ["7B", "8B", "9B"], ["EW", "EW", "EW"], ["RD", "RD"]]
    assert groups[1][2]["drawn"]
    assert len(game_video.hand_groups(tiles, won=False)) == 14


def test_a_face_down_story_shows_the_draws_reads_and_the_hand(tmp_path, monkeypatch):
    folder = read_session(tmp_path)
    monkeypatch.setattr(game_video.base_cam, "tile_boxes", lambda: BOXES)
    monkeypatch.setattr(game_video.face_down_play, "scored_reads", lambda folder: [])
    result = game_video.build(folder, "story", clip_speed=2, out=tmp_path / "fd.mp4")
    assert result["clips"] == 5 and result["reads"] == 2 and result["hand"] == 3
    assert result["read_frames"] == 2 * int(game_video.READ_S * game_video.OUT_FPS)
    assert result["frames"] == result["card_frames"] + result["read_frames"] + 5 * 2      # 4-frame clips at 2x
    assert result["card_frames"] > (game_video.TITLE_S + game_video.HAND_S + game_video.END_S) * game_video.OUT_FPS - 1


def test_a_mat_crop_with_no_tiles_seen_says_the_room_is_in(tmp_path, monkeypatch, capsys):
    """A face-down hand has no scans, so no tile boxes to crop to: the whole frame stays, and it has to say so (the crop is
    what leaves the room out of a published video)."""
    monkeypatch.setattr(game_video.privacy, "blur_faces", lambda frame: (frame, 0))
    desk = game_video.Desk(tmp_path, FACE_DOWN_EVENTS, crop=True)
    desk.prepare(np.zeros((480, 640, 3), np.uint8), FACE_DOWN_EVENTS)
    desk.prepare(np.zeros((480, 640, 3), np.uint8), FACE_DOWN_EVENTS)
    assert capsys.readouterr().out.count("--crop mat") == 1


def test_a_clip_starts_when_its_folder_was_made_even_past_midnight():
    folder = Path("20261104_235900_play")
    assert game_video.clip_start(folder, "001_235930_7B") == game_video.seconds("2026-11-04T23:59:30")
    assert game_video.clip_start(folder, "002_000110_7B") == game_video.seconds("2026-11-05T00:01:10")


def test_the_mat_crop_is_square_around_the_tiles_down_to_the_arm():
    events = [{"type": "detections", "tiles": [{"pixel": [300, 250]}, {"pixel": [500, 300]}]}]
    x0, y0, x1, y1 = game_video.crop_box(events, (480, 848, 3), margin=60)
    assert (x0, x1) == (240, 560) and (y0, y1) == (160, 480)        # tiles wider than the room below them: up
    events = [{"type": "detections", "tiles": [{"pixel": [400, 250]}, {"pixel": [450, 300]}]}]
    x0, y0, x1, y1 = game_video.crop_box(events, (480, 848, 3), margin=60)
    assert (y0, y1) == (190, 480) and x1 - x0 == 290 and x0 <= 340 and x1 >= 510       # narrower: wider about them
    assert game_video.crop_box([], (480, 848, 3)) is None


def test_every_desk_frame_has_its_faces_blurred(tmp_path, monkeypatch):
    monkeypatch.setattr(privacy, "find_faces", lambda image: [(10, 10, 60, 60)])
    folder = tmp_path / "20261104_180000_play"
    (folder / "scans" / "001_register").mkdir(parents=True)
    image = np.random.default_rng(0).integers(0, 255, (480, 848, 3), np.uint8)
    cv2.imwrite(str(folder / "scans" / "001_register" / "f0.jpg"), image)
    desk = game_video.Desk(folder, [], crop=False)
    out = desk.prepare(cv2.imread(str(folder / "scans" / "001_register" / "f0.jpg")), [])
    face = cv2.resize(out, (848, 480))[20:60, 20:60]
    assert face.std() < image[20:60, 20:60].std() / 3            # blurred, not the noise it was


def test_an_unfinished_desk_video_falls_back_to_the_scans(tmp_path, monkeypatch):
    folder = play_sim(tmp_path, monkeypatch, [])
    (folder / desk_video.VIDEO).write_bytes(b"not a video: the game was killed")
    (folder / desk_video.TIMES).write_text(json.dumps({"fps": 2, "times": ["2026-11-04T18:00:00"]}))
    result = game_video.build(folder, "timelapse", out=tmp_path / "t.mp4")
    assert result["desk"].endswith("scans")


def test_a_recorder_that_fails_to_stop_never_stops_the_cleanup(tmp_path, capsys):
    class Broken:
        def stop(self):
            raise OSError("disk full")

    class Session:
        dir = tmp_path
    desk_video.stop(Session(), None, Broken())
    assert "couldn't finish it" in capsys.readouterr().out


def test_a_session_with_nothing_to_show_says_so(tmp_path):
    folder = tmp_path / "20261104_180000_play"
    folder.mkdir()
    (folder / "events.jsonl").write_text(json.dumps(EVENTS[0]) + "\n")
    with pytest.raises(SystemExit, match="no scans, desk video or gripper clips"):
        game_video.build(folder)
