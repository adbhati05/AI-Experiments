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
      <PageHeader title="Performance" subtitle={`The model behind the Detect page (from the fifth training session), measured once on ${dataset.splits.test.images} photos it never saw in training.`} />

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

        <Section title="By damage class" note="Shattered glass, flat tires and broken lamps have sharp, well-defined edges, which makes them easy to localize and therefore score higher. Dents, scratches and cracks fade out at their edges, so both the model and the people who labeled the data have a harder time agreeing on where the damage ends, causing them to score lower. AP50 counts a detection as correct when its box overlaps the labeled box by at least half.">
          <ClassTable />
        </Section>

        <Section title="Choosing the threshold" note="Every detection comes with a confidence score, and the threshold decides how confident the model has to be before a detection is shown. Raising it trades recall for precision, meaning fewer false alarms but more missed damage. A threshold of 0.50 was chosen because it sits near where the two curves cross. The slider on the Detect page moves along this same chart.">
          <ThresholdChart />
        </Section>

        <Section title="Where it goes wrong" note="A confusion matrix compares what the model predicted (rows) against what was actually in the photo (columns), with each column adding up to 100% of one kind of real damage. Most of the errors sit in the bottom row, which is damage the model missed entirely, while almost none come from mistaking one class for another.">
          <ConfusionMatrix />
        </Section>

        <Section title="Precision and recall by class" note="Each curve shows how precision falls as the model is pushed to find more of a class, so a curve that stays high and far to the right belongs to a class the model handles well. The number beside each class in the legend is its AP50, which is the area under its curve.">
          <PrCurves />
        </Section>

        <div className="border-t border-line">
          <Accordion title="Shortcomings">
            <ul className="flex list-disc flex-col gap-2 pl-5">
              <li>
                The model is on the small side: YOLOv8s has {model.parameters_millions} million parameters. Despite being aware of this, I picked it because my MacBook could safely train it in a reasonable amount of time.
              </li>
              <li>
                It often misses damage with soft edges, and at the shipping threshold that works out to about {missedPct('dent')}% of dents,{' '}
                {missedPct('scratch')}% of scratches and {missedPct('crack')}% of cracks going undetected.
              </li>
              <li>
                The labels hold it back too, since where a scratch or dent ends is a judgment call. This makes the training boxes inconsistent and
                means some of the errors counted here are really the model outlining the same damage differently or finding damage nobody labeled.
              </li>
              <li>
                I trained each configuration only once because I was focused on getting this project deployed, so the small differences between
                sessions could come down to chance instead of a real effect.
              </li>
              <li>It has only ever seen one dataset, so I can't say how it holds up on night photos, rain, unusual angles or other cameras.</li>
            </ul>
          </Accordion>
          <Accordion title="Improvements">
            <ul className="flex list-disc flex-col gap-2 pl-5">
              <li>
                Better hardware would help the most, since a dedicated GPU would let me train a larger model at a higher resolution with a full
                batch size, and get through each run much faster.
              </li>
              <li>Training each configuration several times would show which of the differences between sessions are real and which are just noise.</li>
              <li>
                Relabeling the dents, scratches, and cracks under one written rule would give cleaner labels, which raises the ceiling for both
                training the model and scoring it.
              </li>
              <li>More varied photos from different sources would help as well, especially of cracks and other small damage.</li>
              <li>
                Longer training is worth a try, because the larger model hit its best score on its very last epoch, indicating that it hadn't finished
                improving when it reached the 100 epoch limit.
              </li>
            </ul>
          </Accordion>
        </div>
      </div>
    </>
  )
}
