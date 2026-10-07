import { PageHeader } from '../components/PageHeader'
import { Section } from '../components/Section'
import { StatTile } from '../components/StatTile'
import { ClassTable } from '../components/ClassTable'
import { ThresholdChart } from '../components/ThresholdChart'
import { ConfusionMatrix } from '../components/ConfusionMatrix'
import { PrCurves } from '../components/PrCurves'
import { Accordion } from '../components/Accordion'
import { metrics } from '../lib/metrics'

const test = metrics.test.tta
const { model, dataset } = metrics
const cleanFlagged = metrics.clean_cars.after[`conf_${model.conf.toFixed(2)}`].all.images_flagged_pct

const TILES = [
  { label: 'mAP50', value: test.map50, decimals: 3, note: 'boxes that overlap the damage' },
  { label: 'mAP50-95', value: test.map50_95, decimals: 3, note: 'boxes that fit it tightly' },
  { label: 'Precision', value: test.precision, decimals: 3, note: 'detections that are right' },
  { label: 'Recall', value: test.recall, decimals: 3, note: 'damage that is found' },
  { label: 'Clean cars flagged', value: cleanFlagged, decimals: 1, suffix: '%', note: `of ${dataset.clean_eval_images} undamaged cars` },
]

// The configuration of the model that the Detect page runs, laid out like a spec plate.
const SPEC = [
  { label: 'Model', value: model.architecture },
  { label: 'Parameters', value: `${model.parameters_millions}M` },
  { label: 'Input', value: `${model.imgsz} px` },
  { label: 'Threshold', value: model.conf.toFixed(2) },
  { label: 'Test-time augmentation', value: model.tta ? 'on' : 'off' },
  { label: 'Pretrained on', value: model.pretrained_on },
]

// The share of each soft-edged class that the model misses entirely at the shipping threshold, read from the bottom row of the confusion matrix.
const { labels, counts } = metrics.confusion_matrix
const background = labels.indexOf('background')
const missedPct = (name: string) => {
  const column = labels.indexOf(name)
  const total = counts.reduce((sum, row) => sum + row[column], 0)
  return Math.round((counts[background][column] / total) * 100)
}

export default function Performance() {
  return (
    <>
      <PageHeader title="Performance" subtitle={`The model behind the Detect page, measured once on ${dataset.splits.test.images} photos it never saw in training.`} />

      <div className="flex flex-col gap-14">
        <div className="flex flex-col gap-4">
          {/* The 1px gaps between tiles show the line color underneath, which draws the dividers without extra borders. */}
          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-5 [&>*:last-child]:col-span-2 md:[&>*:last-child]:col-span-1">
            {TILES.map((tile) => (
              <StatTile key={tile.label} {...tile} />
            ))}
          </div>

          <dl className="flex flex-wrap gap-x-8 gap-y-3 rounded-lg border border-line px-4 py-3.5">
            {SPEC.map((item) => (
              <div key={item.label}>
                <dt className="eyebrow">{item.label}</dt>
                <dd className="num mt-0.5">{item.value}</dd>
              </div>
            ))}
          </dl>
        </div>

        <Section title="By damage class" note="Glass, tires and lamps have sharp edges and score high. Dents, scratches and cracks fade out at their edges and score lower.">
          <ClassTable />
        </Section>

        <Section title="Choosing the threshold" note="A higher threshold shows fewer false alarms and misses more damage. The slider on the Detect page moves along this chart.">
          <ThresholdChart />
        </Section>

        <Section title="Where it goes wrong" note="Each column is one kind of real damage. Most errors are damage the model missed entirely, in the bottom row, and almost none are one class mistaken for another.">
          <ConfusionMatrix />
        </Section>

        <Section title="Precision and recall by class">
          <PrCurves />
        </Section>

        <div className="border-t border-line">
          <Accordion title="Shortcomings">
            <ul className="flex list-disc flex-col gap-1.5 pl-5">
              <li>
                The model is small. YOLOv8s has {model.parameters_millions} million parameters and was chosen because it trains on my MacBook, not because
                it is the most accurate option.
              </li>
              <li>
                Soft-edged damage is often missed. At the shipping threshold it misses about {missedPct('dent')}% of dents, {missedPct('scratch')}% of
                scratches and {missedPct('crack')}% of cracks.
              </li>
              <li>
                The labels limit it. Where a scratch or dent ends is a judgment call, so the training boxes are inconsistent, and some of the errors
                counted here are the model outlining the same damage differently or finding damage that was never labeled.
              </li>
              <li>Each configuration was trained once (I was focused on getting this project deployed). Small differences between sessions could be chance, not a real effect.</li>
              <li>It has only seen one dataset. Night photos, rain, unusual angles and other cameras are untested.</li>
            </ul>
          </Accordion>
          <Accordion title="Improvements">
            <ul className="flex list-disc flex-col gap-1.5 pl-5">
              <li>Better hardware. A dedicated GPU would allow a larger model, higher resolution at a full batch size and shorter runs.</li>
              <li>Repeated runs. Training each configuration several times would show which differences are real.</li>
              <li>Cleaner labels. Relabeling dents, scratches and cracks under one written rule would raise the ceiling for both training and scoring.</li>
              <li>More varied photos, especially of cracks and other small damage, and from more than one source.</li>
              <li>Longer training. The larger model reached its best score on its final epoch, so it had not finished improving at the 100 epoch limit.</li>
            </ul>
          </Accordion>
        </div>
      </div>
    </>
  )
}
