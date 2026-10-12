import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { addModel } from "./fixtures";

// Local only: TIGER's export is not hosted, and the test needs real conversation. Point
// SPECTRA_SPEAKER_MODELS at a folder holding segmentation-3.0.onnx, voxceleb_ECAPA512.onnx and
// tiger-speech-6s.onnx, and SPECTRA_CONVERSATION at a WAV of two people talking.
const MODELS = process.env["SPECTRA_SPEAKER_MODELS"];
const AUDIO = process.env["SPECTRA_CONVERSATION"];

test("separates a two-person conversation into a layer per person", async ({ page }) => {
  test.skip(!MODELS || !AUDIO, "set SPECTRA_SPEAKER_MODELS and SPECTRA_CONVERSATION");
  test.setTimeout(20 * 60_000);
  await page.goto("/");
  await page.getByTestId("models").click();
  await addModel(page, "speaker-segmentation", join(MODELS ?? "", "segmentation-3.0.onnx"));
  await addModel(page, "speaker-embedding", join(MODELS ?? "", "voxceleb_ECAPA512.onnx"));
  await addModel(page, "speaker-separation", join(MODELS ?? "", "tiger-speech-6s.onnx"));
  await page.getByRole("button", { name: "Close" }).click();

  await page.getByTestId("import").setInputFiles(AUDIO ?? "");
  await expect(page.getByTestId("project-name")).not.toBeEmpty({ timeout: 60_000 });
  const seconds = await page.evaluate(() => {
    const t = document.querySelector('[data-testid="time"]')?.textContent ?? "";
    const [m = 0, s = 0] = (t.split("/")[1] ?? "").trim().split(":").map(Number);
    return m * 60 + s;
  });
  await page.getByTestId("speakers").click();
  await expect(page.getByTestId("speakers-run")).toBeEnabled();
  const dialog = page.getByRole("dialog", { name: "Separate people" });
  console.log("run on:", await dialog.locator("dd").nth(2).innerText());
  const t0 = Date.now();
  await page.getByTestId("speakers-run").click();
  await expect(page.getByTestId("job")).toContainText("Separating crosstalk on the ", {
    timeout: 5 * 60_000,
  });
  console.log("job:", await page.getByTestId("job").innerText());
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 15 * 60_000 });
  const wall = (Date.now() - t0) / 1000;
  console.log(JSON.stringify({ seconds, wall, realTimeFactor: wall / seconds }));
  await expect(page.getByRole("alert")).toHaveCount(0);
  for (const layer of ["person 1", "person 2", "rest"])
    await expect(page.getByTestId(`layer-${layer}`)).toHaveCount(1);
  await expect(page.getByTestId("colours")).toHaveText("Speaker colours");
});
