#!/usr/bin/env python3
"""Put the relative gain beside every absolute delta quoted on a slide."""

p = "build_deck.py"
s = open(p).read()
missed = []


def rep(a, b):
    global s
    if a in s:
        s = s.replace(a, b, 1)
    else:
        missed.append(a[:64])


rep('''foot = txt(s, "head 43.6 → 55.9      pronotum +8.7      spread 7.90 → 3.89",''',
    '''foot = txt(s, "+7.24 mIoU (+12.9%)      head 43.6 → 55.9      "
              "spread 7.90 → 3.89",''')

rep('res = txt(s, "68.54 ± 3.15   →   70.30 ± 1.36", 0.85, 4.85, 11.9, 0.7,',
    'res = txt(s, "68.54 ± 3.15  →  70.30 ± 1.36  (+1.90, +2.8%)",\n'
    '          0.85, 4.85, 11.9, 0.7,')

rep('foot1 = txt(s, "five references average away one bad draw", 0.85, 5.5, 11.9,\n'
    '            0.5, size=15, color=INK, align=PP_ALIGN.CENTER)',
    'foot1 = txt(s, "beetle +9.96 (+15.8%)        butterfly +1.48 (+2.1%)",\n'
    '            0.85, 5.5, 11.9, 0.5, size=15, bold=True, color=INK,\n'
    '            align=PP_ALIGN.CENTER)')

OLD_COST = (
    'foot = txt(s, "cost: cyrbia\'s run-to-run spread rises 6.81 \u2192 10.89",\n'
    "           0.85, 6.3, 11.9, 0.6, size=14, color=INK, align=PP_ALIGN.CENTER)"
)
NEW_COST = (
    'foot = txt(s, "cyrbia 47.40 \u2192 73.27 (+54.6%).  cost: its spread rises "\n'
    '              "6.81 \u2192 10.89",\n'
    "           0.85, 6.3, 11.9, 0.6, size=14, color=INK, align=PP_ALIGN.CENTER)"
)
rep(OLD_COST, NEW_COST)

rep('''foot2 = txt(s, "best arm: TTA on placement-curated references, 77.22",''',
    '''foot2 = txt(s, "best arm: TTA on placement-curated references, 77.22 "
               "(+7.6%)",''')

rep('''foot1 = txt(s, "concept injection, its headline feature, costs ~6 points on both "
               "datasets (0–1 of 20 seeds improved)",''',
    '''foot1 = txt(s, "concept injection costs 5.92 and 6.23 mIoU, which is 8.6% "
               "and 10.0% of what it was added to",''')

rep('''foot2 = txt(s, "the loss stack still pays on it: +0.60 ✱ / +0.58 ✱, above its "
               "value on SAM 2",''',
    '''foot2 = txt(s, "the loss stack still pays on it: +0.60 (+0.8%) and +0.58 "
               "(+1.0%), above its value on SAM 2",''')

rep('''foot = txt(s, "beetle's best 1-shot number:   64.12   →   69.88      "
              "(+5.77 ✱ vs the SAM 2 stack)",''',
    '''foot = txt(s, "beetle's best 1-shot number: 64.12 → 69.88, "
              "+5.77 ✱ (+9.0%) vs the SAM 2 stack",''')

rep('''    ("Orientation TTA is the largest inference-time gain: +5.4, label-free, "
     "no training.", TEAL),''',
    '''    ("Orientation TTA is the largest inference-time gain: +5.44, or +7.6%, "
     "label-free and untrained.", TEAL),''')

rep('''    ("On SAM 3 that ordering reverses: the loss stack and the arch arm are "
     "+3.55 of beetle's +5.77.", COPPER),''',
    '''    ("On SAM 3 that ordering reverses: the loss stack and the arch arm are "
     "+3.55 of beetle's +5.77, which is +9.0%.", COPPER),''')

open(p, "w").write(s)
print("applied:", 10 - len(missed), "missed:", len(missed))
for m in missed:
    print("   MISS:", m)
