// The six damage classes in the order the model was trained with.
export const CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat'] as const

export type ClassName = (typeof CLASS_NAMES)[number]

// Fixed class colors from planning/brainstorming.txt. The same color is used for a class on every box and in every chart.
export const CLASS_COLORS: Record<ClassName, string> = {
  dent: '#3987e5',
  scratch: '#d95926',
  crack: '#199e70',
  'glass shatter': '#e66767',
  'lamp broken': '#9085e9',
  'tire flat': '#d55181',
}

export function classColor(name: string): string {
  return CLASS_COLORS[name as ClassName] ?? '#9fb0a6'
}
