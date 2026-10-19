# DamageDetector web app

The frontend for DamageDetector. It has three pages:

- **Detect** uploads a photo to the API and draws the detections on it, with a confidence slider. Below the results is an optional questionnaire on how the model did.
- **Training** shows the five training sessions and what each one found.
- **Performance** shows how the shipped model scores on the test split.

Built with React, TypeScript, Vite, Tailwind CSS, Recharts and Motion.

## Running it

```
npm install
echo "VITE_API_URL=http://127.0.0.1:8000" > .env.local
npm run dev
```

The site runs at `http://localhost:5173`. The Detect page needs the API running; see the README one folder up. The other two pages work without it.

| Command | What it does |
|---|---|
| `npm run dev` | Starts the dev server |
| `npm run build` | Type-checks and builds to `dist/` |
| `npm run lint` | Runs ESLint |
| `npm run preview` | Serves the built site locally |

## Layout

```
src/
  pages/        Detect, Training, Performance
  components/   the top bar, charts, tables and the upload pieces
  lib/          the API client, shared types and the class colors
  hooks/        the check that the API is awake, and the feedback totals
  data/         metrics.json and the notes for each training session
public/         the logo
```

## Where things are set

- **Colors.** The interface colors are defined once in `src/index.css`, and the six damage class colors in `src/lib/classes.ts`.
- **Numbers on the Training and Performance pages.** They come from `src/data/metrics.json`, which is written by `src/export_metrics.py` in the project root. Rerun that script after retraining instead of editing the file by hand.
- **The API address.** `VITE_API_URL`, in `.env.local` for development and in the host's settings for production.
- **When the feedback counter appears.** `MIN_REPORTS` in `src/components/ReportedStats.tsx`, set to 10. The counter sits in the footer and is hidden below that number or if the API cannot read the totals.

## Deploying

The site is static and deploys to Vercel with the root directory set to this folder.

- `vercel.json` sends every path to `index.html`, so refreshing `/training` or `/performance` works.
- Set `VITE_API_URL` to the deployed API's address.
- Add the site's address to `ALLOWED_ORIGINS` on the API, or the browser will block its requests.
