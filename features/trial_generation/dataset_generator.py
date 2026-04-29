"""Generate the IB experiment dataset.

Each prompt_group contains K pairing facts + K attribute facts and yields K
dataset entries — one per target fact — so every fact in every prompt is
covered by a question. All K entries share the same text_prompt, facts, and
attributes; only target_fact / memory_question / reasoning_question / variants
differ across the K entries.

Structure per entry:
  - K pairing facts:    "A is paired with B"
  - K attribute facts:  "B lives in City"
  - memory_question:    direct recall  (A -> B)            tests I(T;Y) via single-hop
  - reasoning_question: two-hop chain  (A -> B -> City)    tests I(T;Y) via composition
  - causal variants:    target pairing fact removed / corrupted
  - group_id:           shared ID across all K entries from the same prompt
  - target_position:    0-indexed position of the target fact within the prompt group

Output: data/ib_dataset.json
"""

import json
import random
from pathlib import Path

SEED = 20260425
N_GROUPS = 1000   # number of distinct prompt groups; each yields K entries
K_MIN, K_MAX = 4, 6          # number of pairing facts per example
OUTPUT = Path(__file__).resolve().parents[2] / "data" / "ib_dataset.json"

# ---------------------------------------------------------------------------
# Name pools — synthetic, no real-world associations
# ---------------------------------------------------------------------------

PERSON_POOL = [
    "Alen", "Brik", "Cora", "Deni", "Eron", "Fila", "Garo", "Hemi",
    "Ivor", "Juno", "Kelo", "Lome", "Miro", "Nira", "Olen", "Pevi",
    "Quor", "Rema", "Suli", "Toba", "Ulen", "Veki", "Wexa", "Xilo",
    "Yara", "Zuno", "Adri", "Behr", "Calo", "Doke", "Embi", "Falo",
    "Geni", "Huri", "Ipal", "Jelo", "Kuno", "Lera", "Moki", "Nopi",
    "Orva", "Puli", "Qena", "Rilo", "Soma", "Teku", "Umbo", "Voli",
    "Weni", "Xeno", "Yuki", "Zema", "Aspi", "Brok", "Chev", "Druk",
    "Elin", "Frav", "Glor", "Hask", "Ipro", "Jask", "Krev", "Loki",
    "Mevi", "Nask", "Orik", "Plok", "Qrev", "Risp", "Sluk", "Trip",
    "Ursi", "Vrok", "Wask", "Xrip", "Yark", "Zlek", "Avok", "Bren",
    "Crip", "Drev", "Eflo", "Frip", "Gren", "Hrev", "Iflo", "Jrip",
    "Kren", "Lrev", "Mflo", "Nrip", "Oren", "Prev", "Qflo", "Rrip",
    "Sren", "Trev", "Uflo", "Vrip", "Wren", "Xrev", "Yflo", "Zrip",
    "Asko", "Bisk", "Crom", "Dask", "Erov", "Frop", "Glim", "Hosq",
    "Ipom", "Jolm", "Krim", "Lurp",
]
PERSON_POOL = sorted(set(PERSON_POOL))

# Synthetic city names — no real geography
CITY_POOL = [
    "Arvon", "Belso", "Crudo", "Delvi", "Elput", "Fornex", "Gralis",
    "Helto", "Invec", "Jalmo", "Kelso", "Lumex", "Marvix", "Nolto",
    "Oxven", "Palrix", "Quelmo", "Rovnis", "Salpex", "Tuvon", "Ulbex",
    "Vornis", "Welvex", "Xalmo", "Yornix", "Zelpex", "Almov", "Bexol",
    "Colvex", "Delnix", "Elpov", "Folvex", "Galvix", "Holvex", "Ilvox",
]
CITY_POOL = sorted(set(CITY_POOL))

# Largest example needs 2*K persons + 1 corruption replacement (all distinct).
_MAX_PERSONS = 2 * K_MAX + 1
assert len(PERSON_POOL) >= _MAX_PERSONS, f"Person pool too small: need {_MAX_PERSONS}"
assert len(CITY_POOL) >= K_MAX, f"City pool too small: need {K_MAX}"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_prompt(pairs: list[tuple[str, str]], attributes: dict[str, str]) -> str:
    lines = ["Facts:"]
    for a, b in pairs:
        lines.append(f"{a} is paired with {b}")
    for b, city in attributes.items():
        lines.append(f"{b} lives in {city}")
    return "\n".join(lines)


def render_prompt_shuffled(
    pairs: list[tuple[str, str]],
    attributes: dict[str, str],
    rng: random.Random,
) -> str:
    fact_lines = []
    for a, b in pairs:
        fact_lines.append(f"{a} is paired with {b}")
    for b, city in attributes.items():
        fact_lines.append(f"{b} lives in {city}")
    rng.shuffle(fact_lines)
    return "Facts:\n" + "\n".join(fact_lines)


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def make_group(rng: random.Random, group_id: int) -> list[dict]:
    """Generate one prompt group and return K entries, one per target fact.

    All K entries share the same text_prompt, facts, and attributes.
    Each entry targets a different pairing fact, giving 100% fact coverage.
    """
    k = rng.randint(K_MIN, K_MAX)

    # Sample 2k+1 distinct persons: k A-side, k B-side, 1 reserved for corruption.
    persons = rng.sample(PERSON_POOL, 2 * k + 1)
    a_side = persons[:k]
    b_side = persons[k:2 * k]
    corrupt_name = persons[2 * k]

    cities = rng.sample(CITY_POOL, k)
    pairs = list(zip(a_side, b_side))
    attributes = {b: city for b, city in zip(b_side, cities)}

    # Build a single shuffled fact order shared by all K entries.
    all_facts = [(a, b) for a, b in pairs] + [(b, city) for b, city in attributes.items()]
    rng.shuffle(all_facts)

    b_side_set = set(b_side)

    def render_from_pairs(override_pairs: list[tuple[str, str]]) -> str:
        lines = ["Facts:"]
        for fa, fb in all_facts:
            if fb in b_side_set:
                match = next(((a, b) for a, b in override_pairs if a == fa), None)
                if match:
                    lines.append(f"{match[0]} is paired with {match[1]}")
                # omitted if removed
            else:
                lines.append(f"{fa} lives in {fb}")
        return "\n".join(lines)

    text_prompt = render_from_pairs(pairs)

    entries = []
    for target_idx, (target_a, target_b) in enumerate(pairs):
        target_city = attributes[target_b]

        removed_pairs = [p for i, p in enumerate(pairs) if i != target_idx]
        corrupted_pairs = [
            (a, corrupt_name) if a == target_a else (a, b)
            for a, b in pairs
        ]

        entries.append({
            "group_id": group_id,
            "target_position": target_idx,
            "facts": [list(p) for p in pairs],
            "attributes": dict(attributes),
            "text_prompt": text_prompt,
            "memory_question": f"What is {target_a} paired with?",
            "memory_answer": target_b,
            "reasoning_question": f"Where does the person paired with {target_a} live?",
            "reasoning_answer": target_city,
            "target_fact": [target_a, target_b],
            "variants": {
                "removed_prompt": render_from_pairs(removed_pairs),
                "corrupted_prompt": render_from_pairs(corrupted_pairs),
            },
        })

    return entries


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)
    dataset = []
    for group_id in range(N_GROUPS):
        dataset.extend(make_group(rng, group_id))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump(dataset, f, indent=2)
    n_groups = len(set(e["group_id"] for e in dataset))
    print(f"Wrote {len(dataset)} entries ({n_groups} groups) to {OUTPUT}")


if __name__ == "__main__":
    main()
