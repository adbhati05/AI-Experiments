// The hypothesis written down before each training session and what the run showed, taken from planning/logs.txt.
export const SESSION_NOTES: Record<number, { hypothesis: string; result: string }> = {
  1: {
    hypothesis: 'A first run to measure against.',
    result: 'Three classes near ceiling, three weak. The weak ones are missed, not confused with each other.',
  },
  2: {
    hypothesis: 'Higher resolution recovers small damage: crack, dent and scratch improve, the rest stay flat.',
    result: 'Recall rose 0.037 and precision fell 0.024, across all classes. Scratch got worse.',
  },
  3: {
    hypothesis: 'A larger model lifts dent, scratch and crack.',
    result: 'Dent improved. Scratch did not move. Best mAP50-95 so far.',
  },
  4: {
    hypothesis: 'Resolution and capacity stack for a further gain of 0.01 to 0.02.',
    result: 'mAP50-95 fell 0.010 against Session 3. It finds more and localizes worse.',
  },
  5: {
    hypothesis: 'Adding undamaged cars to training cuts false alarms without costing recall.',
    result: 'Clean cars flagged fell from 66% to 14%. Recall on real damage moved by 0.001.',
  },
}
