"""B1K skill annotation → π0.7 subtask sentence.

주석 한 세그먼트(`skill_annotation[i]`)를 π0.7 프롬프트의 `Subtask:` 자리에 들어갈
영어 명령문으로 바꾼다. 어휘는 로컬 18개 태스크 94,357 세그먼트에서 확인한 19개 스킬이 전부다
(`skill_description` 과 `object_id` 는 항상 길이 1 리스트).

객체 이름 정리 규칙:
- 끝의 숫자 토큰 제거 (`apple_pie_234` → apple pie, `half_bell_pepper_213_1` → half bell pepper)
- 숫자 앞의 6글자 모델 id 제거 (`countertop_kelzer_0` → countertop)
- 주석 오타 `hafl` → half, 이중 밑줄 → 공백

memory_prefix 는 "back" / "the other" 두 값만 있다. spatial_prefix 는 객체 리스트와 같은 위치에
정렬된 수식어다 (`["", "low_level"]` = 두 번째 객체의 low level).
"""
from __future__ import annotations

import re

_MODEL_ID = re.compile(r"^[a-z]{6}$")
# 6글자지만 모델 id 가 아닌 실제 단어 — 6글자 토큰이 숫자 앞에 올 때만 지우므로 드물지만 막아 둔다
_REAL_WORDS = {"burner", "bottle", "carton", "cutter", "drawer", "grater", "kettle", "ladder", "napkin",
               "orange", "pepper", "potato", "radish", "saucer", "shaker", "sponge", "tomato", "turkey",
               "walnut", "window", "zipper", "onions", "shelfs", "cheese", "carrot", "celery", "cherry",
               "garlic", "lemons", "melons", "muffin", "noodle", "oyster", "pastry", "peanut", "salmon",
               "sheets", "spoons", "trowel", "fridge", "toaster", "brisket", "bacon"}


_TOOLS = {"carving knife", "parer", "knife", "scissors", "grater"}


def clean_name(obj) -> str:
    if isinstance(obj, list):
        names = [clean_name(o) for o in obj]
        uniq = list(dict.fromkeys(n for n in names if n))
        return " and ".join(uniq)
    s = str(obj).replace("__", "_").strip("_")
    toks = [t for t in s.split("_") if t]
    while toks and toks[-1].isdigit():
        toks.pop()
    if len(toks) >= 2 and _MODEL_ID.match(toks[-1]) and toks[-1] not in _REAL_WORDS:
        toks.pop()
    while toks and toks[-1].isdigit():
        toks.pop()
    toks = ["half" if t == "hafl" else t for t in toks]
    return " ".join(toks)


def _the(name: str) -> str:
    return f"the {name}" if name else "it"


def _spatial(spa: list, i: int) -> str:
    """spatial_prefix 에서 i 번째 객체의 수식어 (없으면 빈 문자열)."""
    if not spa:
        return ""
    row = spa[0] if isinstance(spa[0], list) else spa
    if i < len(row) and row[i]:
        return str(row[i])
    return ""


def _loc(mod: str, obj: str, default_prep: str) -> str:
    """수식어가 붙은 위치 구 — 'on the right side of the burner' 같은 것."""
    o = _the(obj)
    if not mod:
        return f"{default_prep} {o}"
    m = mod.replace("_", " ")
    if mod in ("left", "right"):
        return f"{default_prep} the {m} side of {o}"
    if mod == "center":
        return f"{default_prep} the center of {o}"
    if mod == "to_the_edge_of":
        return f"to the edge of {o}"
    if mod.startswith("layer_"):
        return f"{default_prep} {m} of {o}"
    if mod.endswith("_level"):
        return f"{default_prep} the {m} of {o}"
    return f"{default_prep} the {m} of {o}"


def subtask_sentence(seg: dict) -> str:
    """주석 세그먼트 하나 → 명령문. seg 는 annotation JSON 의 skill_annotation 원소."""
    skill = seg["skill_description"][0]
    objs = seg["object_id"][0]
    if not isinstance(objs, list):
        objs = [objs]
    mem = seg.get("memory_prefix") or []
    mem = mem[0] if mem else ""
    spa = seg.get("spatial_prefix") or []
    names = [clean_name(o) for o in objs]
    a = names[0] if names else ""
    b = names[1] if len(names) > 1 else ""
    c = names[2] if len(names) > 2 else ""
    other = "the other" in mem
    back = "back" in mem
    A = f"the other {a}" if other and a else _the(a)

    if skill == "move to":
        return f"move {'back ' if back else ''}to {A}"
    if skill == "pick up from":
        if b and b != a:
            return f"pick up {A} {_loc(_spatial(spa, 1), b, 'from')}"
        return f"pick up {A}"
    if skill in ("place on", "place in"):
        prep = skill.split()[1]
        return f"place {A}{' back' if back else ''} {_loc(_spatial(spa, 1), b, prep)}"
    if skill == "place on next to":
        side = _spatial(spa, 2)
        rel = f"to the {side} of {_the(c)}" if side in ("left", "right") else f"next to {_the(c)}"
        return f"place {A}{' back' if back else ''} on {_the(b)} {rel}"
    if skill == "push to":
        return f"push {A} {_loc(_spatial(spa, 1), b, 'to')}"
    if skill == "chop":
        # object_id 는 [도구, 대상] 이거나 [대상, 결과물] 이다 (bell_pepper_213 → half_bell_pepper_213_1)
        if b and a in _TOOLS:
            return f"chop {_the(b)} with {_the(a)}"
        return f"chop {A}"
    if skill == "pour":
        # object_id = [[내용물], 담긴 용기, 붓는 곳] 또는 [내용물]
        if len(names) >= 3:
            return f"pour {_the(a)} from {_the(b)} into {_the(c)}"
        return f"pour {A}"
    if skill in ("open door", "close door"):
        verb = skill.split()[0]
        door = _spatial(spa, 0).replace("_", " ")
        if door:
            if "door" not in door:
                door += " door"
            return f"{verb} the {door} of {_the(a)}"
        return f"{verb} {_the(a)}"
    if skill in ("open lid", "close lid"):
        return f"{skill.split()[0]} the lid of {_the(a)}"
    if skill == "turn on switch":
        return f"turn on {_the(a)}"
    if skill == "turn off switch":
        return f"turn off {_the(a)}"
    if skill == "hand over":
        if len(objs) >= 3 and objs[1] in ("left", "right") and objs[2] in ("left", "right"):
            return f"hand over {_the(a)} from the {objs[1]} hand to the {objs[2]} hand"
        return f"hand over {_the(a)} to the other hand"
    if skill == "pull tray":
        lvl = _spatial(spa, 0).replace("_", " ")
        return f"pull out the {lvl + ' ' if lvl else ''}tray of {_the(a)}"
    if skill == "push tray":
        lvl = _spatial(spa, 0).replace("_", " ")
        return f"push in the {lvl + ' ' if lvl else ''}tray of {_the(a)}"
    if skill == "sweep surface":
        return f"sweep the surface of {_the(a)}"
    if skill == "turn to":
        mod = _spatial(spa, 1)
        tgt = "the robot" if b == "robot" else _the(b)
        if mod == "face_away":
            return f"turn {A} to face away from {tgt}"
        return f"turn {A} to face {tgt}"
    # 알 수 없는 스킬 — 원문을 그대로 붙인다 (현재 로컬 주석에는 없음)
    return " ".join([skill] + [_the(n) for n in names if n])


def segment_key(seg: dict) -> tuple:
    """같은 시도의 반복을 잡기 위한 키: (스킬, 객체 원문, 공간 수식어).

    공간 수식어를 넣는 이유: 같은 캐비닛의 왼쪽 문과 오른쪽 문을 차례로 여는 것은 재시도가 아니다.
    """
    return (seg["skill_description"][0], repr(seg["object_id"][0]), repr(seg.get("spatial_prefix") or []))
