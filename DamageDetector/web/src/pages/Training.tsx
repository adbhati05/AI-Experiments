import { useState } from 'react'
import { PageHeader } from '../components/PageHeader'
import { Section } from '../components/Section'
import { SessionTable } from '../components/SessionTable'
import { EpochChart } from '../components/EpochChart'
import { ClassTrends } from '../components/ClassTrends'
import { GeometryTable } from '../components/GeometryTable'
import { CleanCars } from '../components/CleanCars'
import { Accordion } from '../components/Accordion'
import { metrics } from '../lib/metrics'

const { splits, clean_eval_images } = metrics.dataset
const negatives = splits.train.negatives

// Where the data came from and what it was trained on, shown as a compact strip at the top of the page.
const SETUP = [
  { label: 'CarDD', value: `${(splits.train.images - negatives).toLocaleString()} / ${splits.val.images} / ${splits.test.images}`, note: 'train, val and test photos' },
  { label: 'CompCars', value: `${negatives} + ${clean_eval_images}`, note: 'undamaged cars for training and testing' },
  { label: 'Hardware', value: 'M2 Pro, 16 GB', note: 'one laptop, no cloud GPU' },
]

export default function Training() {
  // Shared between the session table and the chart, so hovering either one highlights the same session in both.
  const [active, setActive] = useState<number | null>(null)

  return (
    <>
      <PageHeader title="Training" subtitle="Five runs, one change at a time, each with a prediction written down first." />

      <div className="flex flex-col gap-14">
        <dl className="grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3">
          {SETUP.map((item) => (
            <div key={item.label} className="bg-card px-4 py-3.5">
              <dt className="eyebrow">{item.label}</dt>
              <dd className="num mt-1 text-lg">{item.value}</dd>
              <dd className="text-xs text-muted">{item.note}</dd>
            </div>
          ))}
        </dl>

        <Section title="The five sessions" note="The chart tracks mAP50-95, a score from 0 to 1 for how closely the predicted boxes match the labeled ones, over the course of each run. Select a run for more details on how well each one met expectations.">
          <SessionTable active={active} onActive={setActive} />
          <EpochChart active={active} onActive={setActive} />
        </Section>

        <Section title="What moved, class by class" note="AP50 (average precision, a per-class accuracy score from 0 to 1) on the validation split for each of the five sessions. None of the changes moved the three hard classes by much: dent gained a little, while scratch and crack ended slightly lower than where they started.">
          <ClassTrends />
        </Section>

        <Section title="Why three classes stay hard" note="Larger damage that fills more of its bounding box scores higher, which partly explains why the model struggled with dents, scratches and cracks in particular. Scratches and dents also had by far the most training examples and still scored near the bottom, so the usual expectation that more data yields better performance does not hold here.">
          <GeometryTable />
        </Section>

        <Section title="What the benchmark missed" note={`CarDD contains no undamaged cars, so its metrics could never show how often the model raises a false alarm on a car with nothing wrong with it. Below are the false positives on ${clean_eval_images} clean cars from CompCars that the model never trained on, before and after negative examples were added to the training data.`}>
          <CleanCars />
        </Section>

        <div className="border-t border-line">
          <Accordion title="The bug that cost me a day">
            <p>
              When I first started training, the run looked broken: the losses kept rising, accuracy stayed near zero, and the session got through
              just 7 epochs in a few hours. My first guess was that PyTorch wasn't recognizing the GPU on my MacBook, so I tweaked the code to make
              sure the model was loaded onto it, but nothing changed. To verify nothing else was wrong with the set up, I ran it on the CPU where it learned just fine. 
              After some more digging, it turned out the version of PyTorch I was using (2.4.1) was outdated. I updated to 2.14.1 and the losses dropped,
              accuracy climbed, and epochs finished in minutes instead of hours. The main takeaway is that I forgot to make sure my dependencies were
              up to date, rookie mistake lol.
            </p>
          </Accordion>
          <Accordion title="Future developments I have in mind">
            <ul className="flex list-disc flex-col gap-2 pl-5">
              <li>
                Whole-car photos still set off more false alarms than close-ups do (since every undamaged car I added to training was a close-up) so
                the next training run will add whole-car negatives to close that gap.
              </li>
              <li>
                Right now every class shares a single confidence threshold of 0.50, and I'd like to give each class its own. A flat tire and a
                hairline crack clearly don't deserve the same cutoff. 
              </li>
              <li>
                I also want to try segmentation masks in place of boxes, since a crack only fills about 26% of its bounding box. A
                rectangle is a pretty poor fit for that kind of damage.
              </li>
            </ul>
          </Accordion>
        </div>
      </div>
    </>
  )
}
