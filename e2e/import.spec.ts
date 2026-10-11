import { expect, test } from "@playwright/test";
import { syntheticWav } from "./fixtures";

test("imports a file, plays it, and lists it as a project", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  await page.getByTestId("import").setInputFiles({
    name: "tone-and-hiss.wav",
    mimeType: "audio/wav",
    buffer: syntheticWav(20),
  });

  await expect(page.getByTestId("project-name")).toHaveText("tone-and-hiss", { timeout: 60_000 });
  await expect(page.getByTestId("time")).toContainText("/ 0:20");
  await expect(page.getByTestId("spectrogram")).toBeVisible();

  await page.getByTestId("play").click();
  await expect(page.getByTestId("time")).not.toContainText("0:00.00 /", { timeout: 5_000 });
  await page.waitForTimeout(1_000);
  await page.getByTestId("play").click();
  const played = await page.getByTestId("time").textContent();
  expect(played).toMatch(/^0:0[1-9]\.\d\d \/ 0:20$/);

  await page.getByRole("button", { name: "← Projects" }).click();
  const row = page.getByTestId("project-row").filter({ hasText: "tone-and-hiss" });
  await expect(row).toContainText("0:20 · 44.1 kHz");
  await row.getByRole("button", { name: /tone-and-hiss/ }).click();
  await expect(page.getByTestId("project-name")).toHaveText("tone-and-hiss");
});
