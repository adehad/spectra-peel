import { existsSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { addModel, jobStages, syntheticWav } from "./fixtures";

// Local only: the converted models are not hosted. Build them with reference/, then point
// SPECTRA_STEMS_MODEL at htdemucs-stems.onnx and SPECTRA_FT_MODELS at the folder holding whichever
// htdemucs-ft-<stem>.onnx specialists are built.
const MODEL = process.env["SPECTRA_STEMS_MODEL"];
const FT = process.env["SPECTRA_FT_MODELS"];
const STEMS = ["drums", "bass", "other", "vocals"];

/** Imports a synthetic clip and starts separating it with `best` from specialists. */
async function startStems(page: Page, best: string[], seconds: number) {
  await page
    .getByTestId("import")
    .setInputFiles({ name: "song.wav", mimeType: "audio/wav", buffer: syntheticWav(seconds) });
  await expect(page.getByTestId("project-name")).toHaveText("song", { timeout: 60_000 });
  await page.getByTestId("stems").click();
  for (const stem of best) await page.getByTestId(`best-${stem}`).check();
  await expect(page.getByTestId("stems-run")).toBeEnabled();
  const dialog = page.getByRole("dialog", { name: "Separate stems" });
  console.log("run on:", await dialog.locator("dd").nth(2).innerText());
  await page.getByTestId("stems-run").click();
}

async function expectStems(page: Page) {
  await expect(page.getByRole("alert")).toHaveCount(0);
  for (const stem of [...STEMS, "rest"]) await expect(page.getByTestId(`layer-${stem}`)).toHaveCount(1);
  await expect(page.getByTestId("colours")).toHaveAttribute("aria-pressed", "true");
}

/** Separates a synthetic clip with `best` from specialists and logs the real-time factor. */
async function separate(page: Page, best: string[]) {
  const seconds = Number(process.env["SPECTRA_STEMS_SECONDS"] ?? 20);
  const t0 = Date.now();
  await startStems(page, best, seconds);
  await expect(page.getByTestId("job")).toContainText(" on the ", { timeout: 120_000 });
  console.log("job:", await page.getByTestId("job").innerText());
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 40 * 60_000 });
  const wall = (Date.now() - t0) / 1000;
  console.log(JSON.stringify({ best, seconds, wall, realTimeFactor: wall / seconds }));
  await expectStems(page);
}

async function addModels(page: Page, specialists: string[]) {
  await page.goto("/");
  await page.getByTestId("models").click();
  await addModel(page, "htdemucs-stems", MODEL ?? "");
  for (const stem of specialists)
    await addModel(page, `htdemucs-ft-${stem}`, join(FT ?? "", `htdemucs-ft-${stem}.onnx`));
  await page.getByRole("button", { name: "Close" }).click();
}

test("adds the stems model from a file and separates a clip into four stems", async ({ page }) => {
  test.skip(!MODEL, "set SPECTRA_STEMS_MODEL to the converted htdemucs-stems.onnx");
  test.setTimeout(15 * 60_000);
  await addModels(page, []);
  await separate(page, []);
});

test("takes the chosen stems from their fine-tuned specialists", async ({ page }) => {
  test.skip(!MODEL || !FT, "set SPECTRA_STEMS_MODEL and SPECTRA_FT_MODELS");
  test.setTimeout(45 * 60_000);
  const built = STEMS.filter((s) => existsSync(join(FT ?? "", `htdemucs-ft-${s}.onnx`)));
  await addModels(page, built);
  await separate(page, built);
});

test("resumes after a reload without redoing the passes that finished", async ({ page }) => {
  test.skip(!MODEL || !FT, "set SPECTRA_STEMS_MODEL and SPECTRA_FT_MODELS");
  test.skip(!existsSync(join(FT ?? "", "htdemucs-ft-drums.onnx")), "needs htdemucs-ft-drums.onnx");
  test.setTimeout(30 * 60_000);
  await addModels(page, ["drums"]);
  await startStems(page, ["drums"], 10);
  await expect(page.getByTestId("job")).toContainText("(2 of 2) on the ", { timeout: 20 * 60_000 });
  await page.reload();
  await expect(page.getByTestId("interrupted")).toContainText("Separating stems stopped");
  await page.getByTestId("resume").click();
  const stages = await jobStages(page);
  console.log(stages);
  expect(stages.some((s) => s.includes("(1 of 2)"))).toBe(false);
  await expectStems(page);
});
