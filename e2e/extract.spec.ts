import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { addModel } from "./fixtures";

// Local only: AudioSep's export is not hosted. Point SPECTRA_EXTRACT_MODEL at the file
// reference/audiosep/export_audiosep.py writes, and SPECTRA_EXTRACT_AUDIO at a WAV with speech in it.
// With SPECTRA_OUT set, the Speech and Rest layers are exported there as WAVs for scoring.
const MODEL = process.env["SPECTRA_EXTRACT_MODEL"];
const AUDIO = process.env["SPECTRA_EXTRACT_AUDIO"];
const OUT = process.env["SPECTRA_OUT"];
// A file long enough that a checkpoint is saved before the reload, about two minutes or more.
const LONG = process.env["SPECTRA_EXTRACT_LONG"];

/** Imports `audio`, adds the model when Extract asks for it, and leaves the Extract dialog open. */
async function openExtract(page: Page, audio: string) {
  await page.goto("/");
  await page.getByTestId("import").setInputFiles(audio);
  await expect(page.getByTestId("project-name")).not.toBeEmpty({ timeout: 60_000 });
  await page.getByTestId("extract").click();
  await expect(page.getByRole("dialog", { name: "Add the Extract model" })).toBeVisible();
  await addModel(page, "audiosep-base", MODEL ?? "");
  await page.getByTestId("models-continue").click();
}

/** Solos the `nth` layer called `label` and saves the export as `path`. */
async function exportSolo(page: Page, label: string, nth: number, path: string) {
  const solo = page.getByRole("button", { name: `Solo ${label}` }).nth(nth);
  await solo.click();
  await page.getByTestId("export").click();
  await page.getByRole("radio", { name: /^WAV/ }).check();
  const download = page.waitForEvent("download");
  await page.getByTestId("export-start").click();
  await page.getByTestId("export-download").click();
  await (await download).saveAs(path);
  await page.getByRole("button", { name: "Close" }).click();
  await solo.click();
}

test("extracts speech into its own layer after adding the model it asks for", async ({ page }) => {
  test.skip(!MODEL || !AUDIO, "set SPECTRA_EXTRACT_MODEL and SPECTRA_EXTRACT_AUDIO");
  test.setTimeout(20 * 60_000);
  await openExtract(page, AUDIO ?? "");
  await page.getByTestId("prompt-speech").check();
  const t0 = Date.now();
  await page.getByTestId("extract-run").click();
  await expect(page.getByTestId("job")).toContainText("Speech on the ", { timeout: 120_000 });
  console.log("job:", await page.getByTestId("job").innerText());
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 15 * 60_000 });
  console.log(JSON.stringify({ wall: (Date.now() - t0) / 1000 }));
  await expect(page.getByRole("alert")).toHaveCount(0);
  for (const layer of ["speech", "rest"]) await expect(page.getByTestId(`layer-${layer}`)).toHaveCount(1);
  await expect(page.getByTestId("colours")).toHaveText("Layer colours");

  if (!OUT) return;
  mkdirSync(OUT, { recursive: true });
  for (const layer of ["Speech", "Rest"]) await exportSolo(page, layer, 0, join(OUT, `${layer}.wav`));
});

test("resumes mid-file and writes the same layer as an uninterrupted run", async ({ page }, info) => {
  test.skip(!MODEL || !LONG, "set SPECTRA_EXTRACT_MODEL and SPECTRA_EXTRACT_LONG");
  test.setTimeout(30 * 60_000);
  await openExtract(page, LONG ?? "");
  await page.getByTestId("extract-run").click();
  await expect(page.getByTestId("job")).toContainText(/… ([6-9]\d)%/, { timeout: 15 * 60_000 });
  await page.reload();
  await page.getByTestId("resume").click();
  // The first progress after resuming comes from the checkpoint, not from the start of the file.
  let first = -1;
  while (first < 0) {
    await expect(page.getByRole("alert")).toHaveCount(0);
    const text = (await page.getByTestId("job").textContent({ timeout: 5000 })) ?? "";
    const m = /on the .*… (\d+)%/.exec(text);
    if (m) first = Number(m[1]);
    else await page.waitForTimeout(100);
  }
  console.log({ first });
  expect(first).toBeGreaterThanOrEqual(25);
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 15 * 60_000 });
  await expect(page.getByTestId("layer-speech")).toHaveCount(1);

  await page.getByTestId("extract").click();
  await page.getByTestId("extract-run").click();
  await expect(page.getByTestId("job")).toBeVisible();
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 15 * 60_000 });
  await expect(page.getByTestId("layer-speech")).toHaveCount(2);
  const [resumed, straight] = [info.outputPath("resumed.wav"), info.outputPath("straight.wav")];
  await exportSolo(page, "Speech", 0, resumed);
  await exportSolo(page, "Speech", 1, straight);
  expect(readFileSync(resumed).equals(readFileSync(straight))).toBe(true);
});
