import { expect, test } from "@playwright/test";

test("app is cross-origin isolated", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("isolated")).toHaveText("true");
});
