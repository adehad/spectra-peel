import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { addModel } from "./fixtures";

// Local only: AudioSep's export is not hosted. Point SPECTRA_EXTRACT_MODEL at the file
// reference/audiosep/export_audiosep.py writes, and SPECTRA_EXTRACT_AUDIO at a WAV with speech in it.
// With SPECTRA_OUT set, the Speech and Rest layers are exported there as WAVs for scoring.
const MODEL = process.env["SPECTRA_EXTRACT_MODEL"];
const AUDIO = process.env["SPECTRA_EXTRACT_AUDIO"];
const OUT = process.env["SPECTRA_OUT"];

test("extracts speech into its own layer after adding the model it asks for", async ({ page }) => {
  test.skip(!MODEL || !AUDIO, "set SPECTRA_EXTRACT_MODEL and SPECTRA_EXTRACT_AUDIO");
  test.setTimeout(20 * 60_000);
  await page.goto("/");
  await page.getByTestId("import").setInputFiles(AUDIO ?? "");
  await expect(page.getByTestId("project-name")).not.toBeEmpty({ timeout: 60_000 });
  await page.getByTestId("extract").click();
  await expect(page.getByRole("dialog", { name: "Add the Extract model" })).toBeVisible();
  await addModel(page, "audiosep-base", MODEL ?? "");
  await page.getByTestId("models-continue").click();
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
  for (const layer of ["Speech", "Rest"]) {
    await page.getByRole("button", { name: `Solo ${layer}` }).click();
    await page.getByTestId("export").click();
    await page.getByRole("radio", { name: /^WAV/ }).check();
    const download = page.waitForEvent("download");
    await page.getByTestId("export-start").click();
    await page.getByTestId("export-download").click();
    await (await download).saveAs(join(OUT, `${layer}.wav`));
    await page.getByRole("button", { name: "Close" }).click();
    await page.getByRole("button", { name: `Solo ${layer}` }).click();
  }
});
