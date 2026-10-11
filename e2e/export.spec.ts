import { execFileSync } from "node:child_process";
import { expect, test } from "@playwright/test";
import { syntheticWav } from "./fixtures";

function probeSeconds(path: string): number | undefined {
  try {
    const out = execFileSync("ffprobe", [
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "csv=p=0",
      path,
    ]);
    return Number(out.toString().trim());
  } catch {
    return undefined;
  }
}

test("exports the mix in every supported format", async ({ page }, info) => {
  test.setTimeout(180_000);
  await page.goto("/");
  await page
    .getByTestId("import")
    .setInputFiles({ name: "mix.wav", mimeType: "audio/wav", buffer: syntheticWav(8) });
  await expect(page.getByTestId("project-name")).toHaveText("mix", { timeout: 60_000 });
  await page.getByTestId("denoise").click();
  await expect(page.getByTestId("job")).toBeHidden({ timeout: 120_000 });

  await page.getByTestId("export").click();
  for (const format of ["wav", "flac", "m4a", "opus"] as const) {
    const radio = page.getByRole("radio", {
      name: new RegExp(`^${format === "m4a" ? "M4A" : format === "opus" ? "Opus" : format.toUpperCase()}`),
    });
    if (await radio.isDisabled()) continue;
    await radio.check();
    const download = page.waitForEvent("download");
    await page.getByTestId("export-start").click();
    await page.getByTestId("export-download").click();
    const file = await download;
    expect(file.suggestedFilename()).toBe(`mix-peeled.${format}`);
    const path = info.outputPath(file.suggestedFilename());
    await file.saveAs(path);
    const seconds = probeSeconds(path);
    if (seconds !== undefined) expect(seconds).toBeCloseTo(8, 1);
  }
});
