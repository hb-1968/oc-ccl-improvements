#!/usr/bin/env python3
"""Retone the deck: plain, technical slide text; no em-dashes; no aphorisms."""

SLIDES = [
    # ── 1 · title
    ('"Seven techniques,\\nand which of them\\nactually work"',
     '"Seven techniques,\\nmeasured against\\none baseline"'),
    ('"Hunter Bridges  ·  measured against a fixed baseline, 20 seeded runs each"',
     '"Hunter Bridges  ·  20 seeded runs per arm, paired per seed"'),
    ('"…and what happened when the whole stack moved to two newer models"',
     '"the same stack then carried onto two newer models"'),

    # ── 2 · the task
    ('"one 2-frame video  ·  the masks propagate"',
     '"one 2-frame video  ·  masks propagate from frame 0 to frame 1"'),
    ('"the query is upside down — the prediction follows the support"',
     '"the query is pinned 180°, and the prediction follows the support\'s layout"'),

    # ── 3 · the premise
    ('"Start here. Every technique is a measured delta on this."',
     '"The baseline. Every technique is a measured delta on this."'),
    ('"every technique, judged the same way"', '"how each technique is judged"'),
    ('"the spread is a result, not an error bar"',
     '"run-to-run spread is reported as a result, not an error bar"'),

    # ── 4 · curation
    ('"the train pool\'s own circular std, "\n     "55.6° — never tuned"',
     '"the train pool\'s own circular std, "\n     "55.6°, not tuned"'),

    # ── 5 · the draw
    ('"Curation breaks the species balance. Fix the draw, not the pool.",\n      size=31)',
     '"Curation breaks the species balance. The fix is in the draw.",\n      size=31)'),
    ('"the filter keeps 10/10 of one species, 4/10 of three others"',
     '"the filter keeps 10/10 of one species and 4/10 of three others"'),

    # ── 6 · what they bought
    ('"+7.24 mIoU, and the spread halves."',
     '"Curation and balancing: +7.24 mIoU, spread halved."'),

    # ── 7 · placement
    ('"The same rule, on the defect butterflies actually have."',
     '"The same rule, applied to the defect butterflies have."'),
    ('"Pinned wings are already canonical.\\nThey lie loosely on a "\n             "sheet — what varies is placement, not angle."',
     '"Pinned wings are already canonical.\\nThey lie loosely on a "\n             "sheet, so placement varies rather than angle."'),
    ('"Centroid displacement from the train median.\\nThreshold = "\n             "mean + 1 std — the same discipline."',
     '"Centroid displacement from the train median.\\nThreshold = "\n             "mean + 1 std, the same discipline."'),
    ('"the two criteria fired on disjoint subspecies — two mechanisms, "\n              "not one lucky knob"',
     '"the two criteria fired on disjoint subspecies, each where its "\n              "diagnostic predicted"'),

    # ── 8 · shots
    ('"Five references instead of one — and where that stops helping.",\n      size=31)',
     '"Five references instead of one, and where that stops helping.",\n      size=31)'),
    ('"five supports average away one bad draw"',
     '"five references average away one bad draw"'),
    ('"so butterfly\'s problem is not the support"',
     '"butterfly gains 1.48; its defect is on the query side"'),

    # ── 9 · the flip
    ('"0.00\\n\\nwhat the copy baseline scores\\non those specimens —\\n"\n              "no model involved"',
     '"0.00\\n\\nwhat the copy baseline scores\\non those specimens,\\n"\n              "with no model involved"'),
    ('"curation +3.56   ·   five shots +2.85   ·   the level never "\n              "moved"',
     '"curation +3.56   ·   five shots +2.85   ·   the level did not "\n              "move"'),

    # ── 10 · displacement
    ('"the top quartile already beats the target — the deficit is "\n              "all in the tail"',
     '"the top quartile already beats the target; the deficit is in "\n              "the tail"'),

    # ── 11 · TTA section
    ('"The largest single gain in the project — and the one that needed the "\n       "least machinery."',
     '"The largest single gain in the project, and the least machinery.\\n"\n       "No training and no extra labels."'),

    # ── 12 · TTA mechanism 1
    ('"pass 1 — as given"', '"pass 1, as given"'),
    ('"pass 2 — support rotated 180°"', '"pass 2, support rotated 180°"'),
    ('"the query — never touched"', '"the query, never touched"'),
    ('("The rotated variant is cached apart", "@r180 — or pass 2 gets the "\n     "wrong embeddings"),',
     '("The rotated variant is cached apart", "@r180, or pass 2 gets the "\n     "wrong embeddings"),'),

    # ── 13 · TTA mechanism 2
    ('"both orientations cached — pass 2 is a hit"',
     '"both orientations cached, so pass 2 is a hit"'),
    ('"it chooses between two predictions — it cannot improve either "\n              "one"',
     '"it chooses between two predictions; it cannot improve either one"'),

    # ── 14 · TTA results
    ('"the cost: cyrbia\'s run-to-run spread rises 6.81 → 10.89"',
     '"cost: cyrbia\'s run-to-run spread rises 6.81 → 10.89"'),

    # ── 15 · the 2x2
    ('"technique 6: let the five supports span both orientations — "\n               "the memory bank selects, it does not blend"',
     '"technique 6 is support-side: the five references span both "\n               "orientations, and the memory bank selects rather than blends"'),
    ('"two mechanisms, one defect  —  interaction −4.66"',
     '"support-side +4.02      query-side +5.44      both +4.80      '
     'interaction −4.66"'),

    # ── 16 · loss section
    ('"OC-CCL — a cycle-consistency fine-tune. The most machinery, and the "\n       "least return."',
     '"OC-CCL, a cycle-consistency fine-tune.\\nThe most machinery in the "\n       "project."'),

    # ── 17 · the cycle
    ('"Track to the query, then track back and check you land where you "\n         "started."',
     '"Track to the query, then track back and check the reconstruction."'),
    ('"no new labels — the reference\'s own mask is the target"',
     '"no new labels required; the reference\'s own mask is the target"'),

    # ── 18 · preconditions
    ('"severs the graph — nothing downstream reaches the forward leg."',
     '"severs the graph, so nothing downstream reaches the forward leg."'),
    ('"Each pays only where its own objective still has room."',
     '"Each pays where its own objective has room."'),
    ('"butterfly silhouette had room → +0.85 to +1.25        "\n             "beetle silhouette already 90.8 → nothing"',
     '"butterfly silhouette had room → +0.85 to +1.25        "\n             "beetle silhouette at 90.8 → no effect"'),

    # ── 19 · SCNP
    ('"tried twice — on the merged silhouette, then on the per-trait "\n              "cycle where “same class” finally means same trait"',
     '"tried twice: on the merged silhouette, then on the per-trait cycle "\n              "where “same class” means same trait"'),

    # ── 21 · the pot
    ('"the rest is regions the tracker never proposes — no loss on "\n              "proposed logits reaches them.   remember that sentence"',
     '"the rest is regions the tracker never proposes, which no loss on "\n              "proposed logits reaches"'),

    # ── 24 · gates
    ('"~5 of butterfly\'s last 6 points sit below the silhouette "\n              "ceiling — which is why the last act changes the model"',
     '"~5 of butterfly\'s last 6 points sit below the silhouette ceiling"'),

    # ── 25 · the swaps
    ('"A concept LVLM steering a retrained SAM 2.1 tracker — identity "\n             "reasoning instead of pixel memory."',
     '"A concept LVLM steering a retrained SAM 2.1 tracker: identity "\n             "reasoning rather than pixel memory."'),
    ('"The same tracker design, one generation on — so every technique "\n             "in this talk ports unchanged."',
     '"The same tracker design one generation on, so every technique "\n             "here ports unchanged."'),
    ('"and one new arm: LoRA on the memory attention itself — the "\n               "proposal-forming module no arm on any family had trained"',
     '"one new arm: LoRA on the memory attention, the proposal-forming "\n               "module no arm on any family had trained"'),

    # ── 26 · SeC
    ('"its headline feature — concept injection — costs ~6 points on "\n               "both datasets, 0–1 of 20 seeds improved"',
     '"concept injection, its headline feature, costs ~6 points on both "\n               "datasets (0–1 of 20 seeds improved)"'),
    ('"but our loss stack still pays on it: +0.60 ✱ / +0.58 ✱ — "\n               "better than on its home family"',
     '"the loss stack still pays on it: +0.60 ✱ / +0.58 ✱, above its "\n               "value on SAM 2"'),

    # ── 27 · SAM 3
    ('"The second swap pays — and the loss work with it."',
     '"The second swap pays, and the loss work with it."'),
    ('"beetle\'s best 1-shot number:   64.12   →   69.88      "\n              "(+5.77 ✱ against the SAM 2 stack)"',
     '"beetle\'s best 1-shot number:   64.12   →   69.88      "\n              "(+5.77 ✱ vs the SAM 2 stack)"'),

    # ── 28 · the ceiling
    ('"the arch arm trains the memory attention — the module that "\n               "decides what gets proposed at all"',
     '"the arch arm trains the memory attention, the module that decides "\n               "what is proposed"'),
    ('"so it reaches the error mass a loss on proposed logits "\n               "provably cannot"',
     '"so it reaches error mass that a loss on proposed logits cannot"'),

    # ── 29 · projection
    ('"bands frozen from measured components — shot lift, TTA\'s growth "\n              "with shot, the stack, the arch arm"',
     '"bands assembled from measured components: shot lift, TTA\'s growth "\n              "with shot, the stack, the arch arm"'),
    ('"a central landing would sit ≈ +3 over the paper\'s best beetle "\n               "row — the size of their own headline claim"',
     '"a central landing sits ≈ +3 over the paper\'s best beetle row"'),

    # ── 30 · close
    ('("On one tracker, everything that paid changed what the model is shown — "\n     "not what it optimises.", PAPER),',
     '("On SAM 2, every technique that paid changed what the model is shown, "\n     "not what it optimises.", PAPER),'),
    ('("Orientation TTA is the best inference-time return: +5.4, label-free, "\n     "no training.", TEAL),',
     '("Orientation TTA is the largest inference-time gain: +5.4, label-free, "\n     "no training.", TEAL),'),
    ('("Change the model and that inverts: the loss stack and the architecture "\n     "arm are worth +3.55 of beetle\'s +5.77.", COPPER),',
     '("On SAM 3 that ordering reverses: the loss stack and the arch arm are "\n     "+3.55 of beetle\'s +5.77.", COPPER),'),
    ('("Because they moved the ceiling, not the assignment. Butterfly\'s "\n     "residual survived both swaps — it was never capacity.", PAPER),',
     '("They moved the ceiling rather than the assignment. Butterfly\'s "\n     "residual survived both swaps.", PAPER),'),
]

FIGURES = [
    ('"silhouette − mIoU\\n(the correlate we quoted)"',
     '"silhouette − mIoU\\n(a correlate)"'),
    ('"oracle − argmax\\n(the real ceiling)"',
     '"oracle − argmax\\n(the measured ceiling)"'),
    ('''"74.95 — what cyrbia's never-flipped\\n"
                       "specimens score: the ceiling this\\n"
                       "intervention can reach"''',
     '''"74.95: what cyrbia's never-flipped\\n"
                       "specimens score, i.e. the ceiling\\n"
                       "for this intervention"'''),
    ('"20 draws over 12 species — equal probability is not equal "\n                 "representation"',
     '"20 draws over 12 species: equal probability is not equal "\n                 "representation"'),
    ('"BEETLE — every rung significant, and they compose"',
     '"BEETLE: every rung significant, and they compose"'),
    ('"BUTTERFLY — parity, both ns"', '"BUTTERFLY: parity, both ns"'),
]


def apply(path, pairs):
    s = open(path).read()
    missed = []
    for a, b in pairs:
        if a in s:
            s = s.replace(a, b, 1)
        else:
            missed.append(a[:64])
    open(path, "w").write(s)
    return missed


if __name__ == "__main__":
    for path, pairs in (("build_deck.py", SLIDES),
                        ("make_talk_figures.py", FIGURES)):
        missed = apply(path, pairs)
        print(f"{path}: {len(pairs) - len(missed)} applied, {len(missed)} missed")
        for m in missed:
            print("   MISS:", m)
