#!/usr/bin/env python3
"""Geometry pass after the restyle: clear the rule, equalise boxes, reflow."""

p = "build_deck.py"
s = open(p).read()
missed = []


def rep(a, b):
    global s
    if a in s:
        s = s.replace(a, b, 1)
    else:
        missed.append(a[:70])


# slide 2 · captions were sitting on the title rule
rep("    t = txt(s, lab, x, 1.53, w, 0.3, size=11.5, bold=True, color=MUTED,",
    "    t = txt(s, lab, x, 1.74, w, 0.3, size=11.5, bold=True, color=INK,")
rep("    p = pic(s, f, x, 1.9, h=2.4)", "    p = pic(s, f, x, 2.06, h=2.4)")

# slide 1 · labels centred in their boxes
rep("    txt(s, lab, x, 4.08, 3.77, 0.4, size=14, bold=True, font=HEAD, color=INK,",
    "    txt(s, lab, x, 4.15, 3.77, 0.4, size=14, bold=True, font=HEAD, color=INK,")

# slide 4 · no orphaned last word
rep('''r1 = txt(s, "20 seeded draws   ·   same draws both arms   ·   paired per seed"
            "   ·   CI   ·   band frozen first",''',
    '''r1 = txt(s, "20 draws / same draws both arms / paired per seed / CI / "
            "band frozen first",''')

# slide 31 · the close gets the title and rule every other slide has
rep('''kicker(s, "IN ONE LINE EACH", y=1.4)''',
    '''kicker(s, "IN ONE LINE EACH")
title(s, "Where it lands")''')
rep('''    t = txt(s, l, 0.9, 2.15 + i * 1.25, 11.5, 1.1, size=16,
            bold=(i in (1, 2)), font=HEAD, color=c, line_spacing=1.2)''',
    '''    t = txt(s, l, 0.72, 2.1 + i * 1.15, 11.9, 1.0, size=15,
            bold=(i in (1, 2)), font=HEAD, color=INK, line_spacing=1.25)''')

# slides 8, 19, 26 · equal boxes on a common grid
for a, b in (("c1 = card(s, 0.85, 1.72, 5.75, 2.8, fill=WASH)",
              "c1 = card(s, 0.7, 1.98, 5.85, 2.8)"),
             ("c2 = card(s, 7.0, 1.72, 5.5, 2.8, fill=WASH)",
              "c2 = card(s, 6.75, 1.98, 5.85, 2.8)"),
             ("c1 = card(s, 0.85, 1.7, 5.7, 4.15, fill=WASH)",
              "c1 = card(s, 0.7, 1.98, 5.85, 4.0)"),
             ("c2 = card(s, 6.9, 1.7, 5.6, 4.15, fill=WASH)",
              "c2 = card(s, 6.75, 1.98, 5.85, 4.0)"),
             ("c1 = card(s, 0.85, 1.9, 5.9, 2.6, fill=WASH)",
              "c1 = card(s, 0.7, 1.98, 5.85, 2.6)"),
             ("c2 = card(s, 7.0, 1.9, 5.5, 2.6, fill=WASH)",
              "c2 = card(s, 6.75, 1.98, 5.85, 2.6)")):
    rep(a, b)

# slide 26 · text follows its box, and the last line does not orphan
rep('t1 = txt(s, "SeC-4B", 1.2, 2.12, 5.2, 0.6, size=22, bold=True, font=HEAD,',
    't1 = txt(s, "SeC-4B", 1.0, 2.2, 5.2, 0.6, size=22, bold=True, font=HEAD,')
rep('''t1b = txt(s, "A concept LVLM steering a retrained SAM 2.1 tracker: identity "
             "reasoning rather than pixel memory.", 1.2, 2.85, 5.2, 1.4,
          size=13, color=INK, line_spacing=1.3)''',
    '''t1b = txt(s, "A concept LVLM steering a retrained SAM 2.1\\ntracker: identity "
             "reasoning, not pixel memory.", 1.0, 3.0, 5.3, 1.4,
          size=12, color=INK, line_spacing=1.35)''')
rep('t2 = txt(s, "SAM 3", 7.35, 2.12, 4.8, 0.6, size=22, bold=True, font=HEAD,',
    't2 = txt(s, "SAM 3", 7.05, 2.2, 4.8, 0.6, size=22, bold=True, font=HEAD,')
rep('''t2b = txt(s, "The same tracker design one generation on, so every technique "
             "here ports unchanged.", 7.35, 2.85, 4.8, 1.4, size=13,
          color=INK, line_spacing=1.3)''',
    '''t2b = txt(s, "The same tracker design one generation on,\\nso every technique "
             "here ports unchanged.", 7.05, 3.0, 5.3, 1.4, size=12,
          color=INK, line_spacing=1.35)''')
rep('''extra = txt(s, "one new arm: LoRA on the memory attention, the proposal-forming "
               "module no arm on any family had trained",
            0.85, 5.65, 11.9, 0.9, size=14, color=COPPER,
            align=PP_ALIGN.CENTER, line_spacing=1.2)''',
    '''extra = txt(s, "one new arm: LoRA on the memory attention, the module that "
               "forms proposals", 0.7, 5.8, 11.9, 0.5, size=14, color=INK,
            align=PP_ALIGN.CENTER)''')

# slide 32 · boxes inside the rule
rep("    c = card(s, x, 1.98, 3.95, 4.05)", "    c = card(s, x, 1.98, 3.85, 4.05)")
rep("    x = 0.75 + i * 4.15", "    x = 0.72 + i * 4.03")

# slide 33 · first row clear of the rule
rep("    y = 1.62 + i * 0.86", "    y = 2.0 + i * 0.8")
rep('''foot = txt(s, "memoised per dataset instance above canonical_pool, never "
              "inside it, so the leakage test cannot pass vacuously",
           0.8, 6.85, 11.9, 0.5, size=12.5, italic=True, color=COPPER)''',
    '''foot = txt(s, "memoised above canonical_pool, never inside it, so the leakage "
              "test cannot pass vacuously",
           0.7, 6.9, 11.9, 0.4, size=12.5, italic=True, color=INK)''')

# slide 34 · leading between a wrapped item and its sub-line
rep("    y = 1.7 + (i // 2) * 1.15", "    y = 2.0 + (i // 2) * 1.28")
rep("    t2 = txt(s, why, x + 0.4, y + 0.46, 5.5, 0.4, size=11.5, color=MUTED)",
    "    t2 = txt(s, why, x + 0.4, y + 0.62, 5.5, 0.4, size=11.5, color=INK)")
rep('''rec = txt(s, "recorded in every bundle: criteria, criteria_applied, pool_size, "
             "train_size, rejection counts by reason, threshold_deg, "
             "train_circular_mean_deg, train_resultant_R, threshold_rule, "
             "placement_threshold_px",
          0.8, 5.5, 11.9, 1.1, size=12.5, color=INK, line_spacing=1.25)''',
    '''rec = txt(s, "recorded per run: criteria, criteria_applied, pool_size, "
             "train_size, rejections by reason, threshold_deg, "
             "train_circular_mean_deg, train_resultant_R, threshold_rule, "
             "placement_threshold_px",
          0.7, 5.9, 11.9, 0.9, size=12, color=INK, line_spacing=1.25)''')

# slide 10 · the callout reads as one sentence
rep('''note = txt(s, "0.00\\n\\nwhat the copy baseline scores\\non those specimens,\\n"
              "with no model involved", 9.15, 2.5, 3.5, 3.0, size=14, color=CRIMSON,
           bold=True, line_spacing=1.3)''',
    '''note = txt(s, "0.00 is what the copy baseline\\nscores on those specimens,\\n"
              "with no model involved", 9.15, 2.6, 3.6, 1.6, size=13,
           color=INK, bold=True, line_spacing=1.35)''')

open(p, "w").write(s)
print("applied:", 22 - len(missed), "missed:", len(missed))
for m in missed:
    print("   MISS:", m)
