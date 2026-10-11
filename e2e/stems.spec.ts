import { expect, test } from "@playwright/test";
import { syntheticWav } from "./fixtures";

// Local only: needs the 345 MB converted model, which is not hosted. Build it with reference/ and point
// SPECTRA_STEMS_MODEL at it.
const MODEL = process.env["SPECTRA_STEMS_MODEL"];
test.skip(!MODEL, "set SPECTRA_STEMS_MODEL to the converted htdemucs-stems.onnx");

test("adds the stems model from a file and separates a clip into four stems", async ({ page }) => {
  test.setTimeout(15 * 60_000);
  await page.goto("/");
  await page.getByTestId("models").click();
  await page
    .getByTestId("model-htdemucs-stems")
    .locator('input[type="file"]')
    .setInputFiles(MODEL ?? "");
  await expect(page.getByTestId("model-htdemucs-stems")).toContainText("Ready", { timeout: 120_000 });
  await page.getByRole("button", { name: "Close" }).click();

  const seconds = Number(process.env["SPECTRA_STEMS_SECONDS"] ?? 20);
  await page
    .getByTestId("import")
    .setInputFiles({ name: "song.wav", mimeType: "audio/wav", buffer: syntheticWav(seconds) });
  await expect(page.getByTestId("project-name")).toHaveText("song", { timeout: 60_000 });
  await page.getByTestId("stems").click();
  const t0 = Date.now();
  await page.getByTestId("stems-run").click();
  await expect(page.getByTestId("job")).toBeVisible();
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 14 * 60_000 });
  const wall = (Date.now() - t0) / 1000;
  console.log(JSON.stringify({ seconds, wall, realTimeFactor: wall / seconds }));
  for (const stem of ["drums", "bass", "other", "vocals", "rest"]) {
    await expect(page.getByTestId(`layer-${stem}`)).toHaveCount(1);
  }
});
