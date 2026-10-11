import { expect, test } from "@playwright/test";
import { syntheticWav } from "./fixtures";

test("denoises the whole file, then a selection, and undoes it", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/");
  await page.getByTestId("import").setInputFiles({
    name: "hiss.wav",
    mimeType: "audio/wav",
    buffer: syntheticWav(20),
  });
  await expect(page.getByTestId("project-name")).toHaveText("hiss", { timeout: 60_000 });

  await page.getByTestId("denoise").click();
  await expect(page.getByTestId("job")).toBeVisible();
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 120_000 });
  await expect(page.getByTestId("layer-denoised")).toHaveCount(1);
  await expect(page.getByTestId("layer-removed")).toHaveCount(1);
  await expect(page.getByTestId("view-layer")).toHaveValue(/-denoised$/);

  const box = await page.getByTestId("spectrogram").boundingBox();
  if (!box) throw new Error("no spectrogram");
  await page.keyboard.down("Shift");
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await expect(page.getByTestId("denoise")).toHaveText("Denoise selection");
  await page.getByTestId("denoise").click();
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 120_000 });
  await expect(page.getByTestId("layer-denoised")).toHaveCount(2);

  await page.getByTestId("play").click();
  await expect(page.getByTestId("time")).not.toContainText("0:00.00 /", { timeout: 5_000 });
  await page.getByTestId("play").click();

  await page.getByRole("button", { name: "Undo" }).click();
  await expect(page.getByTestId("layer-denoised")).toHaveCount(1);
});
