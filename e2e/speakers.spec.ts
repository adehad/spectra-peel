import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { addModel, jobStages } from "./fixtures";

// Local only: TIGER's export is not hosted, and the test needs real conversation. Point
// SPECTRA_SPEAKER_MODELS at a folder holding segmentation-3.0.onnx, voxceleb_ECAPA512.onnx and
// tiger-speech-6s.onnx, and SPECTRA_CONVERSATION at a WAV of two people talking.
const MODELS = process.env["SPECTRA_SPEAKER_MODELS"];
const AUDIO = process.env["SPECTRA_CONVERSATION"];

/** Imports the conversation and asks for People, adding the models it asks for on the way. */
async function openConversation(page: Page): Promise<void> {
  await page.goto("/");
  await page.getByTestId("import").setInputFiles(AUDIO ?? "");
  await expect(page.getByTestId("project-name")).not.toBeEmpty({ timeout: 60_000 });
  await page.getByTestId("speakers").click();
  const panel = page.getByRole("dialog", { name: "Add the People models" });
  await expect(panel.locator('[data-testid^="model-"]')).toHaveCount(3);
  await expect(page.getByTestId("models-continue")).toBeDisabled();
  await addModel(page, "speaker-segmentation", join(MODELS ?? "", "segmentation-3.0.onnx"));
  await addModel(page, "speaker-embedding", join(MODELS ?? "", "voxceleb_ECAPA512.onnx"));
  await addModel(page, "speaker-separation", join(MODELS ?? "", "tiger-speech-6s.onnx"));
  await page.getByTestId("models-continue").click();
}

const expectPeople = async (page: Page) => {
  await expect(page.getByRole("alert")).toHaveCount(0);
  for (const layer of ["person 1", "person 2", "rest"])
    await expect(page.getByTestId(`layer-${layer}`)).toHaveCount(1);
};

test("separates a two-person conversation into a layer per person", async ({ page }) => {
  test.skip(!MODELS || !AUDIO, "set SPECTRA_SPEAKER_MODELS and SPECTRA_CONVERSATION");
  test.setTimeout(20 * 60_000);
  await openConversation(page);
  const seconds = await page.evaluate(() => {
    const t = document.querySelector('[data-testid="time"]')?.textContent ?? "";
    const [m = 0, s = 0] = (t.split("/")[1] ?? "").trim().split(":").map(Number);
    return m * 60 + s;
  });
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
  await expectPeople(page);
  await expect(page.getByTestId("colours")).toHaveText("Speaker colours");
});

test("reopens the project after a reload and resumes from the saved crosstalk", async ({ page }) => {
  test.skip(!MODELS || !AUDIO, "set SPECTRA_SPEAKER_MODELS and SPECTRA_CONVERSATION");
  test.setTimeout(20 * 60_000);
  await openConversation(page);
  await page.getByTestId("speakers-run").click();
  await expect(page.getByTestId("job")).toContainText(/Separating crosstalk on the .*… (5[5-9]|[67]\d)%/, {
    timeout: 10 * 60_000,
  });
  await page.reload();
  await expect(page.getByTestId("interrupted")).toContainText("Separating people stopped");
  await expect(page.getByTestId("speakers")).toBeDisabled();
  await page.getByTestId("resume").click();
  const stages = await jobStages(page);
  console.log(stages);
  expect(stages.some((s) => s.startsWith("Finding who speaks when"))).toBe(false);
  await expectPeople(page);
  await expect(page.getByTestId("interrupted")).toHaveCount(0);
});
