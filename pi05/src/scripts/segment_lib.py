"""BEHAVIOR-1K annotation -> 세그먼트 + 자연어 지시문 변환.

에피소드(평균 수천 프레임)를 skill_annotation의 frame_duration으로 잘라
pi0.5가 다룰 수 있는 길이의 세그먼트로 만들고, 각 세그먼트에
skill_description + object_id + prefix 로부터 영어 지시문을 생성한다.
"""
import re

# scene 객체 id에 붙는 6자 랜덤 해시. 데이터셋이 고정이므로 명시적으로 나열한다.
# (자동 판별은 'bell_pepper_214'의 'pepper'처럼 실제 단어를 해시로 오인한다.)
MODEL_HASHES = {
    "dszchb", "egwapq", "kelzer", "kelker", "awvzkn", "fancyy", "lkxmne",
    "ffitak", "petcxr", "lkklqs", "mjvqii", "hjjxmi", "abzvij", "byvbuc",
    "kxcpgy", "zqjkvm", "tynnnw",
}
unknown_hashes = set()   # 미등록 해시 발견 시 여기 기록 (검증용)

def clean_object(o):
    """object_id -> 사람이 읽는 이름.

    fridge_dszchb_0        -> fridge
    bell_pepper_214        -> bell pepper      (pepper를 해시로 오인하지 않음)
    half_vidalia_onion_80_1-> half vidalia onion
    diced__head_cabbage    -> diced head cabbage
    """
    if not isinstance(o, str):
        return None
    s = o
    if s.startswith("hafl_"):          # 원본 데이터 오타 1건
        s = "half_" + s[5:]
    # half_<물체>_<번호>_<반쪽번호> 형태의 꼬리 숫자들을 제거
    while True:
        m = re.match(r"^(.*?)_(\d+)$", s)
        if not m:
            break
        s = m.group(1)
    # 남은 꼬리가 모델 해시면 제거
    m = re.match(r"^(.*?)_([a-z]{6})$", s)
    if m:
        if m.group(2) in MODEL_HASHES:
            s = m.group(1)
        else:
            unknown_hashes.add(m.group(2))
    return re.sub(r"_+", " ", s).strip()


def _flat(x):
    """중첩 리스트를 평탄화 (pour의 첫 인자가 [['diced__head_cabbage'], ...] 형태)."""
    out = []
    if isinstance(x, list):
        for i in x:
            out.extend(_flat(i))
    elif x is not None:
        out.append(x)
    return out


def _slot_name(slot):
    """하나의 인자 슬롯 -> 이름 하나.

    슬롯은 문자열이거나, 같은 물체의 여러 인스턴스 리스트다.
    (예: pour의 첫 인자 ['bacon_209',...,'bacon_214'] -> "bacon")
    평탄화해서 붙이면 슬롯이 밀리므로 반드시 슬롯 단위로 접는다.
    """
    names = [clean_object(o) for o in _flat(slot)]
    names = [n for n in names if n]
    if not names:
        return None
    uniq = list(dict.fromkeys(names))
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) == 2:
        return f"{uniq[0]} and {uniq[1]}"
    return ", ".join(uniq[:-1]) + f", and {uniq[-1]}"


def _objs(skill):
    """object_id -> 슬롯별 이름 리스트. 슬롯 개수가 곧 스킬의 인자 개수."""
    grps = skill.get("object_id") or []
    g = grps[0] if grps and isinstance(grps[0], list) else grps
    if not isinstance(g, list):
        return []
    return [_slot_name(slot) for slot in g]


def _prefixes(skill, key):
    return [p for p in _flat(skill.get(key) or []) if isinstance(p, str) and p]


LEVEL = {
    "low_level": "the low shelf", "middle_level": "the middle shelf",
    "high_level": "the high shelf",
    "layer_3": "the 3rd shelf", "layer_4": "the 4th shelf", "layer_5": "the 5th shelf",
}
DOOR = {
    "left_door": "the left door", "right_door": "the right door",
    "right_upper_door": "the upper right door", "right_lower_door": "the lower right door",
    "left_upper_door": "the upper left door", "left_lower_door": "the lower left door",
}

def make_instruction(skill):
    """하나의 skill_annotation -> 영어 지시문. 만들 수 없으면 None."""
    desc = (skill.get("skill_description") or [None])[0]
    if not desc:
        return None
    o = _objs(skill)
    sp = _prefixes(skill, "spatial_prefix")
    mem = _prefixes(skill, "memory_prefix")

    def the(i):
        return f"the {o[i]}" if i < len(o) and o[i] else None

    back = "back " if "back" in mem else ""
    other = "the other " if "the other" in mem else None

    def obj(i):
        if i >= len(o) or not o[i]:
            return None
        return (other + o[i]) if other else f"the {o[i]}"

    lvl = next((LEVEL[p] for p in sp if p in LEVEL), None)
    door = next((DOOR[p] for p in sp if p in DOOR), None)
    side = next((p for p in sp if p in ("left", "right", "center", "in_front_of",
                                        "away", "to_the_edge_of")), None)
    side_txt = {"in_front_of": "in front of", "to_the_edge_of": "the edge of",
                "center": "the center of", "away": "away from"}.get(side, side)

    if desc == "move to":
        return f"move {back}to {obj(0)}" if obj(0) else None
    if desc == "pick up from":
        if not (obj(0) and the(1)): return None
        s = f"pick up {obj(0)} from {the(1)}"
        return s + (f" on {lvl}" if lvl else "")
    if desc in ("place on", "place in"):
        if not (obj(0) and the(1)): return None
        prep = "on" if desc == "place on" else "in"
        s = f"put {obj(0)} {back}{prep} {the(1)}"
        return s + (f" on {lvl}" if lvl else "")
    if desc in ("place on next to", "place in next to"):
        if not (obj(0) and the(1) and the(2)): return None
        prep = "on" if desc == "place on next to" else "in"
        rel = f"to the {side_txt} of" if side in ("left", "right") else (side_txt or "next to")
        return f"put {obj(0)} {back}{prep} {the(1)} {rel} {the(2)}"
    if desc == "push to":
        if not (obj(0) and the(1)): return None
        if side == "away":
            return f"push {obj(0)} away from {the(1)}"
        return f"push {obj(0)} to {side_txt or 'the center of'} {the(1)}"
    if desc == "chop":
        # object_id = [도구, 대상]
        if not (the(0) and obj(1)): return None
        return f"chop {obj(1)} with {the(0)}"
    if desc == "pour":
        if len(o) >= 3 and o[0] and o[1] and o[2]:
            return f"pour {obj(0)} from {the(1)} into {the(2)}"
        return f"pour out {obj(0)}" if obj(0) else None
    if desc in ("open door", "close door"):
        if not the(0): return None
        v = desc.split()[0]
        return f"{v} {door or 'the door'} of {the(0)}"
    if desc in ("open lid", "close lid"):
        return f"{desc.split()[0]} the lid of {the(0)}" if the(0) else None
    if desc in ("turn on switch", "turn off switch"):
        return f"turn {desc.split()[1]} {the(0)}" if the(0) else None
    if desc in ("pull tray", "push tray"):
        v = "pull out" if desc.startswith("pull") else "push in"
        return f"{v} the tray of {the(0)}" if the(0) else None
    if desc == "hand over":
        if len(o) >= 3 and o[1] and o[2]:
            return f"hand over {obj(0)} from the {o[1]} hand to the {o[2]} hand"
        return f"hand over {obj(0)}" if obj(0) else None
    if desc == "turn to":
        if not (obj(0) and the(1)): return None
        if "face_away" in sp:
            return f"turn {obj(0)} to face away from {the(1)}"
        return f"turn {obj(0)} to face {the(1)}"
    if desc == "tip over":
        return f"tip over {obj(0)}" if obj(0) else None
    if desc in ("sweep surface", "wipe hard", "spray", "press"):
        return f"{desc.split()[0]} {the(0)}" if the(0) else None
    return None


def spans(skill):
    """frame_duration을 [(start,end), ...] 로 정규화. [a,b] / [[a,b],...] 양쪽 처리."""
    fd = skill.get("frame_duration") or []
    if not fd:
        return []
    raw = fd if isinstance(fd[0], list) else [fd]
    out = []
    for sp in raw:
        if len(sp) == 2 and all(isinstance(v, (int, float)) for v in sp) and sp[1] > sp[0]:
            out.append((int(sp[0]), int(sp[1])))
    return out
