import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..", "..");
const names = ["result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx"];
const previewDir = path.join(root, "working", "artifact-q234", "previews");
await fs.mkdir(previewDir, { recursive: true });

const report = [];
for (const name of names) {
  const input = await FileBlob.load(path.join(root, "results", name));
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheets = workbook.worksheets.items;
  const item = { file: name, sheets: [] };
  for (const sheet of sheets) {
    const used = sheet.getUsedRange(true);
    const table = await workbook.inspect({
      kind: "table",
      sheetId: sheet.name,
      range: sheet.name === "计划购电量" || sheet.name === "调整购电量" ? "A1:L6" : "A1:F12",
      include: "values,formulas",
      tableMaxRows: 12,
      tableMaxCols: 12,
      maxChars: 8000,
    });
    const errors = await workbook.inspect({
      kind: "match",
      sheetId: sheet.name,
      searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
      options: { useRegex: true, maxResults: 20 },
      summary: "formula error scan",
      maxChars: 2000,
    });
    const preview = await workbook.render({
      sheetName: sheet.name,
      range: sheet.name === "计划购电量" || sheet.name === "调整购电量" ? "A1:L6" : "A1:F12",
      scale: 1.2,
      format: "png",
    });
    const png = `${name.replace(".xlsx", "")}_${sheet.name}.png`;
    await fs.writeFile(path.join(previewDir, png), new Uint8Array(await preview.arrayBuffer()));
    item.sheets.push({ name: sheet.name, usedAddress: used?.address ?? null, table: table.ndjson, errors: errors.ndjson, preview: png });
  }
  report.push(item);
}
await fs.writeFile(path.join(root, "working", "artifact-q234", "verification.json"), JSON.stringify(report, null, 2), "utf8");
console.log(JSON.stringify(report.map(x => ({ file: x.file, sheets: x.sheets.map(s => ({name: s.name, usedAddress: s.usedAddress, preview: s.preview})) })), null, 2));
