import { expect, test } from "@playwright/test";

test("app is cross-origin isolated", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("isolated")).toHaveText("true");
});

test("spike runs DeepFilterNet3 on wasm", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/spike/");
  const result = await page.evaluate(() => window.runSpike({ model: "dfn3", ep: "wasm", seconds: 10 }));
  expect(result.error).toBeUndefined();
  expect(result.environment["crossOriginIsolated"]).toBe(true);
  expect(result.report?.epUsed).toBe("wasm");
});
