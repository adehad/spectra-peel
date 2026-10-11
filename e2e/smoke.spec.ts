import { expect, test } from "@playwright/test";

test("app is cross-origin isolated", async ({ page }) => {
  await page.goto("/");
  expect(await page.evaluate(() => globalThis.crossOriginIsolated)).toBe(true);
  await expect(page.getByText("Import audio or video")).toBeVisible();
});
