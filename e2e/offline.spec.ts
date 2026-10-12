import { expect, test } from "@playwright/test";
import { denoise, syntheticWav } from "./fixtures";

test("works offline once installed, still cross-origin isolated", async ({ page, context }) => {
  test.setTimeout(180_000);
  await page.goto("/");
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
  });
  await page.reload();
  await page.waitForFunction(() => navigator.serviceWorker.controller !== null);

  await context.setOffline(true);
  await page.reload();
  await expect(page.getByText("Import audio or video")).toBeVisible();
  expect(await page.evaluate(() => globalThis.crossOriginIsolated)).toBe(true);

  await page
    .getByTestId("import")
    .setInputFiles({ name: "offline.wav", mimeType: "audio/wav", buffer: syntheticWav(6) });
  await expect(page.getByTestId("project-name")).toHaveText("offline", { timeout: 60_000 });
  await denoise(page);
  await expect(page.getByTestId("layer-denoised")).toHaveCount(1);
});
