import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workspace = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const dataDir = path.join(workspace, "working", "prepared_data");
const outputDir = path.join(workspace, "outputs", "01a08ae9-b1f2-7cb1-8533-3abdadca9c5b");
const qaDir = path.join(workspace, "working", "qa_preprocessing");
const outputPath = path.join(outputDir, "数据预处理结果.xlsx");
const fontName = "Arial";

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });

async function readJson(name) {
  return JSON.parse(await fs.readFile(path.join(dataDir, name), "utf8"));
}

const a1 = await readJson("附件1处理后.json");
const actual = await readJson("全年实际处理后.json");
const forecast = await readJson("光伏预测处理后.json");
const descriptions = await readJson("描述统计.json");
const missingSummary = await readJson("缺失值汇总.json");
const anomalyDetails = await readJson("异常值明细.json");
const monthlyAnomaly = await readJson("月度异常汇总.json");
const histograms = await readJson("直方图数据.json");
const standardization = await readJson("标准化参数.json");
const qualityChecks = await readJson("质量检查.json");
const auditSummary = await readJson("审计摘要.json");

const workbook = Workbook.create();

function colLetter(index) {
  let n = index + 1;
  let result = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    result = String.fromCharCode(65 + rem) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

function parseLocalDate(value) {
  if (value == null || value === "") return null;
  const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2}))?$/);
  if (!match) return value;
  const utcMilliseconds = Date.UTC(
    Number(match[1]),
    Number(match[2]) - 1,
    Number(match[3]),
    Number(match[4] || 0),
    Number(match[5] || 0),
    Number(match[6] || 0),
  );
  return utcMilliseconds / 86400000 + 25569;
}

function displayValue(column, value) {
  if (typeof value === "boolean") return value ? "是" : "否";
  if (value == null) return null;
  if (["日期", "日期时间", "发布时间", "目标时间"].includes(column)) return parseLocalDate(value);
  return value;
}

function titleBlock(sheet, title, subtitle, lastCol) {
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: fontName, size: 16, bold: true, color: "#1F1F1F" };
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange("A2").format.font = { name: fontName, size: 10, italic: true, color: "#595959" };
  sheet.getRange(`A3:${lastCol}3`).format.borders = {
    bottom: { style: "thin", color: "#A6A6A6" },
  };
}

function styleHeader(range) {
  range.format = {
    fill: "#1F4E78",
    font: { name: fontName, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D9E2F3" },
  };
  range.format.rowHeight = 30;
}

function styleSection(range) {
  range.format = {
    fill: "#D9EAF7",
    font: { name: fontName, size: 11, bold: true, color: "#1F1F1F" },
    borders: { preset: "outside", style: "thin", color: "#9EADBA" },
  };
}

function writeMatrix(sheet, startRow, startCol, matrix, batchSize = 2000) {
  for (let offset = 0; offset < matrix.length; offset += batchSize) {
    const batch = matrix.slice(offset, offset + batchSize);
    sheet.getRangeByIndexes(startRow + offset, startCol, batch.length, batch[0].length).values = batch;
  }
}

function writeRecords(sheet, headerRowIndex, records, columns) {
  const headers = columns.map((c) => c.label);
  sheet.getRangeByIndexes(headerRowIndex, 0, 1, headers.length).values = [headers];
  styleHeader(sheet.getRangeByIndexes(headerRowIndex, 0, 1, headers.length));
  const matrix = records.map((row) => columns.map((c) => displayValue(c.key, row[c.key])));
  if (matrix.length) writeMatrix(sheet, headerRowIndex + 1, 0, matrix);
  const lastRow = headerRowIndex + matrix.length;
  const used = sheet.getRangeByIndexes(0, 0, lastRow + 1, headers.length);
  if (matrix.length <= 5000) {
    used.format.font = { name: fontName, size: 10, color: "#262626" };
  }
  styleHeader(sheet.getRangeByIndexes(headerRowIndex, 0, 1, headers.length));
  sheet.freezePanes.freezeRows(headerRowIndex + 1);
  return { firstDataRow: headerRowIndex + 2, lastExcelRow: lastRow + 1, lastCol: colLetter(headers.length - 1) };
}

function setColumnWidths(sheet, columns) {
  columns.forEach((column, index) => {
    sheet.getRange(`${colLetter(index)}:${colLetter(index)}`).format.columnWidth = column.width || 14;
  });
}

function applyFormats(sheet, columns, firstDataRow, lastExcelRow) {
  columns.forEach((column, index) => {
    const letter = colLetter(index);
    const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastExcelRow}`);
    if (column.format) range.format.numberFormat = column.format;
    if (column.align) range.format.horizontalAlignment = column.align;
    if (column.wrap) range.format.wrapText = true;
    if (column.key.includes("异常") || column.key.includes("结构性填充")) {
      range.conditionalFormats.add("containsText", {
        text: "是",
        format: { fill: "#FCE4D6", font: { color: "#C00000", bold: true } },
      });
    }
  });
}

function addDataSheet(name, title, subtitle, records, columns, existingSheet = null) {
  const sheet = existingSheet || workbook.worksheets.add(name);
  titleBlock(sheet, title, subtitle, colLetter(columns.length - 1));
  const bounds = writeRecords(sheet, 3, records, columns);
  setColumnWidths(sheet, columns);
  applyFormats(sheet, columns, bounds.firstDataRow, bounds.lastExcelRow);
  return { sheet, ...bounds };
}

// 说明与汇总
const summary = workbook.worksheets.add("说明与汇总");
titleBlock(summary, "数据预处理结果", "高教社杯数学建模竞赛C题附件数据；原始文件未被覆盖", "H");
summary.getRange("A5:C5").values = [["审计指标", "结果", "处理结论"]];
styleHeader(summary.getRange("A5:C5"));
const summaryRows = [
  ["附件1单日记录数", auditSummary["附件1记录数"], "144个10分钟时段，完整"],
  ["全年实际记录数", auditSummary["全年实际记录数"], "365天×144时段，完整"],
  ["光伏预测长表记录数", auditSummary["光伏预测记录数"], "365天×4次发布×24小时，完整"],
  ["数值字段缺失数", 0, "无需均值、线性插值或近邻填充"],
  ["附件3结构性日期空白", auditSummary["结构性日期填充数"], "同一日期后3行省略重复日期，已向下填充"],
  ["年末无对应实际光伏", auditSummary["年末无对应实际光伏数"], "预测目标超出实际数据范围，保留为空"],
  ["四分位距候选异常数", auditSummary["异常值明细数"], "未发现负值或断裂，全部保留并标记"],
  ["被删除或修正的数值", 0, "避免删除真实负荷、光伏或价格峰值"],
];
writeMatrix(summary, 5, 0, summaryRows);
summary.getRange("A15:C15").values = [["质量检查", "结果", "结论"]];
styleHeader(summary.getRange("A15:C15"));
writeMatrix(summary, 15, 0, qualityChecks.map((r) => [r["检查项"], `${r["结果"]}（预期${r["预期"]}）`, r["结论"]]));
summary.getRange("A23:C23").values = [["处理环节", "采用方案", "依据"]];
styleHeader(summary.getRange("A23:C23"));
const methodRows = [
  ["缺失值", "实际数值不填充；附件3日期向下填充", "只有合并单元格造成的结构性日期空白，没有数值缺失"],
  ["异常值", "四分位距检测，保留并标记", "光伏含大量零值且各变量具有明显时序结构，不适合仅用全局标准分数删除"],
  ["功率转电量", "10分钟功率除以6", "功率乘以1/6小时得到千瓦时"],
  ["标准化", "仅预测用字段采用一月份参数的标准分数", "近邻和支持向量方法受量纲影响，优化模型必须保留物理单位"],
  ["时间特征", "加入日内和年内正弦、余弦编码", "避免23:50与0:00在数值距离上被误判为很远"],
];
writeMatrix(summary, 23, 0, methodRows);
summary.getRange("A31:C31").values = [["建模使用说明", null, null]];
styleSection(summary.getRange("A31:C31"));
summary.getRange("A32:C36").values = [
  ["说明1", null, "优化模型直接使用千瓦、千瓦时和元每千瓦时列，不进行标准化。"],
  ["说明2", null, "预测模型若采用近邻或支持向量方法，可使用一月份拟合的标准分数；滚动训练时应只用当前日期以前的数据重新估计参数。"],
  ["说明3", null, "四分位距异常标记表示统计上少见，不等同于录入错误；当前数据未发现物理非法值，因此没有删除或修正。"],
  ["说明4", null, "附件3年末36条记录没有对应实际光伏，仅因预测目标超出附件2范围，不参与预测误差评价。"],
  ["说明5", null, "所有日期、发布时间和目标时间均已展开为独立字段，可直接进行滚动回测。"],
];
summary.getRange("A32:C36").format.wrapText = true;
summary.getRange("A32:C36").format.rowHeight = 28;
summary.getRange("A32:C36").format.font = { name: fontName, size: 10 };
summary.getRange("A:A").format.columnWidth = 24;
summary.getRange("B:B").format.columnWidth = 20;
summary.getRange("C:C").format.columnWidth = 58;
summary.getRange("D:H").format.columnWidth = 12;

// 缺失值汇总
const missingColumns = [
  { key: "数据集", label: "数据集", width: 22 },
  { key: "字段", label: "字段", width: 32 },
  { key: "原始缺失数", label: "原始缺失数", width: 14, format: "#,##0" },
  { key: "缺失性质", label: "缺失性质", width: 28, wrap: true },
  { key: "处理方法", label: "处理方法", width: 45, wrap: true },
  { key: "处理后缺失数", label: "处理后缺失数", width: 16, format: "#,##0" },
];
const missingSheetInfo = addDataSheet(
  "缺失值汇总",
  "缺失值检查与处理",
  "数值字段完整；仅处理结构性日期空白，并保留超出评价范围的年末空值",
  missingSummary,
  missingColumns,
);
const missingSheet = missingSheetInfo.sheet;
const strategyStart = missingSheetInfo.lastExcelRow + 3;
missingSheet.getRange(`A${strategyStart}:F${strategyStart}`).values = [["未来若出现缺失", "优先方法", "适用条件", "不采用方法", "理由", "是否本次执行"]];
styleHeader(missingSheet.getRange(`A${strategyStart}:F${strategyStart}`));
const missingStrategies = [
  ["内部短缺口", "线性插值", "连续缺失不超过3个10分钟时段，前后数据存在", "全局均值填充", "时间序列局部连续，均值会破坏峰谷形态", "否，本次无数值缺失"],
  ["较长但非整日缺口", "相似日近邻填充", "可找到同季节、同星期和相似曲线日期", "简单线性外推", "长缺口线性外推容易低估峰值", "否，本次无数值缺失"],
  ["连续缺失超过2小时或整日", "回查原始数据；无法补充时剔除该日训练样本", "缺失范围过大", "强制生成完整曲线", "大范围填充会制造虚假数据", "否"],
  ["序列首尾缺失", "相似日近邻填充", "没有双侧相邻点，无法线性插值", "均值填充", "均值不能反映具体时刻和季节", "否"],
];
writeMatrix(missingSheet, strategyStart, 0, missingStrategies);
missingSheet.getRange(`A${strategyStart + 1}:F${strategyStart + missingStrategies.length}`).format.wrapText = true;
missingSheet.getRange(`A${strategyStart + 1}:F${strategyStart + missingStrategies.length}`).format.rowHeight = 34;

// 异常值汇总
const descriptionColumns = [
  { key: "数据集", label: "数据集", width: 20 },
  { key: "变量", label: "变量", width: 22 },
  { key: "样本数", label: "样本数", width: 12, format: "#,##0" },
  { key: "缺失数", label: "缺失数", width: 10, format: "#,##0" },
  { key: "最小值", label: "最小值", width: 14, format: "#,##0.0000" },
  { key: "下四分位数", label: "下四分位数", width: 14, format: "#,##0.0000" },
  { key: "中位数", label: "中位数", width: 14, format: "#,##0.0000" },
  { key: "上四分位数", label: "上四分位数", width: 14, format: "#,##0.0000" },
  { key: "最大值", label: "最大值", width: 14, format: "#,##0.0000" },
  { key: "均值", label: "均值", width: 14, format: "#,##0.0000" },
  { key: "标准差", label: "标准差", width: 14, format: "#,##0.0000" },
  { key: "偏度", label: "偏度", width: 11, format: "0.0000" },
  { key: "四分位距异常数", label: "候选异常数", width: 14, format: "#,##0" },
  { key: "异常率", label: "异常率", width: 11, format: "0.00%" },
  { key: "负值数", label: "负值数", width: 10, format: "#,##0" },
  { key: "处理动作", label: "处理动作", width: 20 },
];
addDataSheet(
  "异常值汇总",
  "异常值检测结果",
  "附件1采用全局四分位距；全年数据采用同月同一时刻四分位距；检测值均未自动删除",
  descriptions,
  descriptionColumns,
);

// 异常值明细
const anomalyColumns = [
  { key: "数据集", label: "数据集", width: 20 },
  { key: "变量", label: "变量", width: 22 },
  { key: "日期时间或定位", label: "日期时间或定位", width: 34 },
  { key: "数值", label: "数值", width: 15, format: "#,##0.0000" },
  { key: "检测下界", label: "检测下界", width: 15, format: "#,##0.0000" },
  { key: "检测上界", label: "检测上界", width: 15, format: "#,##0.0000" },
  { key: "检测规则", label: "检测规则", width: 30 },
  { key: "处理动作", label: "处理动作", width: 16 },
  { key: "理由", label: "理由", width: 58, wrap: true },
];
addDataSheet(
  "异常值明细",
  "候选异常值明细",
  "异常标记仅表示相对同月同一时刻较少见；未发现负功率、负价格或时间断裂",
  anomalyDetails,
  anomalyColumns,
);

// 附件1处理后
const a1Columns = [
  { key: "时段序号", label: "时段序号", width: 10, format: "0" },
  { key: "时刻", label: "时刻", width: 10 },
  { key: "电价", label: "电价（元每千瓦时）", width: 18, format: "0.0000" },
  { key: "小区负载功率_千瓦", label: "小区负载功率（千瓦）", width: 20, format: "#,##0.0000" },
  { key: "光伏预测功率_千瓦", label: "光伏预测功率（千瓦）", width: 20, format: "#,##0.0000" },
  { key: "负荷电量_千瓦时", label: "负荷电量（千瓦时）", width: 20, format: "#,##0.0000" },
  { key: "光伏预测电量_千瓦时", label: "光伏预测电量（千瓦时）", width: 22, format: "#,##0.0000" },
  { key: "电价异常", label: "电价异常", width: 11 },
  { key: "小区负载功率异常", label: "负荷异常", width: 11 },
  { key: "光伏预测功率异常", label: "光伏异常", width: 11 },
];
const a1Info = addDataSheet(
  "附件1处理后",
  "附件1单日数据",
  "原始数值完整；功率已换算为每10分钟电量；异常检测未发现超出全局四分位距的数值",
  a1,
  a1Columns,
);
a1Info.sheet.getRange(`F${a1Info.firstDataRow}`).formulas = [[`=D${a1Info.firstDataRow}/6`]];
a1Info.sheet.getRange(`F${a1Info.firstDataRow}:F${a1Info.lastExcelRow}`).fillDown();
a1Info.sheet.getRange(`G${a1Info.firstDataRow}`).formulas = [[`=E${a1Info.firstDataRow}/6`]];
a1Info.sheet.getRange(`G${a1Info.firstDataRow}:G${a1Info.lastExcelRow}`).fillDown();

// 全年实际数据处理后
// 先创建被公式引用的标准化参数工作表。
const standardizationPlaceholder = workbook.worksheets.add("标准化参数");
const actualColumns = [
  { key: "日期时间", label: "日期时间", width: 19, format: "yyyy-mm-dd hh:mm" },
  { key: "日期", label: "所属日期", width: 12, format: "yyyy-mm-dd" },
  { key: "时刻", label: "时刻", width: 9 },
  { key: "时段序号", label: "时段序号", width: 10, format: "0" },
  { key: "小区负载功率_千瓦", label: "小区负载功率（千瓦）", width: 20, format: "#,##0.0000" },
  { key: "光伏实际功率_千瓦", label: "光伏实际功率（千瓦）", width: 20, format: "#,##0.0000" },
  { key: "波动电价_元每千瓦时", label: "波动电价（元每千瓦时）", width: 22, format: "0.0000" },
  { key: "月份", label: "月份", width: 8, format: "0" },
  { key: "星期", label: "星期", width: 8, format: "0" },
  { key: "年内日序", label: "年内日序", width: 10, format: "0" },
  { key: "负荷电量_千瓦时", label: "负荷电量（千瓦时）", width: 20, format: "#,##0.0000" },
  { key: "光伏实际电量_千瓦时", label: "光伏实际电量（千瓦时）", width: 20, format: "#,##0.0000" },
  { key: "时刻正弦", label: "时刻正弦", width: 12, format: "0.0000" },
  { key: "时刻余弦", label: "时刻余弦", width: 12, format: "0.0000" },
  { key: "年内正弦", label: "年内正弦", width: 12, format: "0.0000" },
  { key: "年内余弦", label: "年内余弦", width: 12, format: "0.0000" },
  { key: "小区负载功率异常", label: "负荷异常", width: 11 },
  { key: "光伏实际功率异常", label: "光伏异常", width: 11 },
  { key: "波动电价异常", label: "电价异常", width: 11 },
  { key: "负荷标准分数_一月参数", label: "负荷标准分数（一月参数）", width: 22, format: "0.0000" },
  { key: "光伏标准分数_一月参数", label: "光伏标准分数（一月参数）", width: 22, format: "0.0000" },
  { key: "电价标准分数_一月参数", label: "电价标准分数（一月参数）", width: 22, format: "0.0000" },
];
const actualInfo = addDataSheet(
  "全年实际处理后",
  "全年负荷、光伏实际功率与波动电价",
  "共52560个10分钟记录；原始数值保持不变；异常值保留并增加标记",
  actual,
  actualColumns,
);

// 光伏预测处理后
const forecastColumns = [
  { key: "日期", label: "发布日期", width: 12, format: "yyyy-mm-dd" },
  { key: "预报时刻", label: "原预报时刻", width: 12 },
  { key: "日期结构性填充", label: "日期结构性填充", width: 16 },
  { key: "光伏预测功率_千瓦", label: "光伏预测功率（千瓦）", width: 20, format: "#,##0.0000" },
  { key: "预测步长小时", label: "预测步长（小时）", width: 17, format: "0" },
  { key: "发布小时", label: "发布小时", width: 11, format: "0" },
  { key: "发布时间", label: "发布时间", width: 19, format: "yyyy-mm-dd hh:mm" },
  { key: "目标时间", label: "目标时间", width: 19, format: "yyyy-mm-dd hh:mm" },
  { key: "月份", label: "月份", width: 8, format: "0" },
  { key: "光伏预测功率异常", label: "预测值异常", width: 12 },
  { key: "光伏实际功率_千瓦", label: "对应实际光伏（千瓦）", width: 20, format: "#,##0.0000" },
  { key: "预测误差_千瓦", label: "预测误差（千瓦）", width: 18, format: "#,##0.0000" },
  { key: "绝对误差_千瓦", label: "绝对误差（千瓦）", width: 18, format: "#,##0.0000" },
];
const forecastInfo = addDataSheet(
  "光伏预测处理后",
  "附件3光伏预测长表",
  "日期已向下填充；每次发布的24小时预测已展开；目标时间已匹配实际光伏用于误差评价",
  forecast,
  forecastColumns,
);

// 标准化参数
const standardizationColumns = [
  { key: "变量", label: "变量", width: 22 },
  { key: "拟合区间", label: "拟合区间", width: 24 },
  { key: "样本数", label: "样本数", width: 12, format: "#,##0" },
  { key: "均值", label: "均值", width: 15, format: "#,##0.000000" },
  { key: "标准差", label: "标准差", width: 15, format: "#,##0.000000" },
  { key: "最小值", label: "最小值", width: 15, format: "#,##0.000000" },
  { key: "最大值", label: "最大值", width: 15, format: "#,##0.000000" },
  { key: "标准化用途", label: "标准化用途", width: 58, wrap: true },
];
const standardInfo = addDataSheet(
  "标准化参数",
  "预测模型标准化参数",
  "参数仅由2025年1月初始训练窗口拟合，避免使用2月至12月未来信息",
  standardization,
  standardizationColumns,
  standardizationPlaceholder,
);
standardInfo.sheet.getRange("A10:D10").values = [["转换", "公式", "是否执行", "说明"]];
styleHeader(standardInfo.sheet.getRange("A10:D10"));
writeMatrix(standardInfo.sheet, 10, 0, [
  ["标准分数", "（原值－训练集均值）÷训练集标准差", "执行", "用于近邻、支持向量等距离型预测方法；全年处理表已提供结果"],
  ["最小—最大归一化", "（原值－训练集最小值）÷（训练集最大值－训练集最小值）", "不执行", "极端值会显著改变区间，且本题优化模型需要保留物理量纲"],
  ["功率转电量", "功率×10÷60", "执行", "对应10分钟时段，等价于功率除以6"],
  ["周期时间编码", "日内或年内位置分别转换为正弦和余弦", "执行", "保持时间周期首尾相邻关系"],
]);
standardInfo.sheet.getRange("A11:D14").format.wrapText = true;
standardInfo.sheet.getRange("A11:D14").format.rowHeight = 54;
standardInfo.sheet.getRange("A:A").format.columnWidth = 22;
standardInfo.sheet.getRange("B:B").format.columnWidth = 40;
standardInfo.sheet.getRange("C:C").format.columnWidth = 14;
standardInfo.sheet.getRange("D:D").format.columnWidth = 56;

// 数据补充建议
const supplements = workbook.worksheets.add("数据补充建议");
titleBlock(supplements, "数据与规则补充建议", "区分必须明确的建模口径与可选外部数据；当前预处理未引入外部数据", "F");
supplements.getRange("A5:F5").values = [["类别", "项目", "影响问题", "为什么需要", "无补充时建议假设", "建议检验"]];
styleHeader(supplements.getRange("A5:F5"));
const supplementRows = [
  ["必须明确", "90%效率是单向效率还是循环总效率", "全部问题", "直接影响储电量状态转移和可用放电量", "主模型按充、放电单向效率均为90%；另做循环总效率90%的敏感性分析", "比较总费用和日末储电量"],
  ["必须明确", "是否允许弃光、是否允许向外网售电", "全部问题", "决定光伏过剩时的能量平衡方式", "允许无成本弃光，不允许向外网售电", "报告弃光量并检验是否大量出现"],
  ["必须明确", "外网最大购电功率", "全部问题", "若无限购电，紧急购电只受价格约束", "题目未给上限时设为不受限，并在论文中说明", "增加有限购电上限进行敏感性分析"],
  ["必须明确", "每天0:00可获得哪些负荷、光伏和价格信息", "问题二、四", "防止直接使用当天实际值造成未来信息泄露", "仅使用当前时刻以前的数据预测未来", "采用逐日滚动回测"],
  ["必须明确", "调整购电费用的结算口径", "问题三、四", "关系到计划费与调整费是否重复计算", "分别核算计划量、减少量违约费和增加量加价，并列出费用恒等式", "逐时段费用对账"],
  ["必须明确", "问题二至四的期末储电量规则", "问题二至四", "只有问题一明确要求每日首尾相等", "保持跨日连续；全年末设置回到初始值或加入末端价值", "比较两种期末规则"],
  ["必须明确", "额外预测的获取成本", "问题三", "判断是否增加预测时刻需要经济代价", "先按零获取成本计算理论价值，再对假设成本做敏感性分析", "计算边际信息价值"],
  ["可选补充", "气温、云量、辐照度和降水", "问题二至四", "可提高负荷和光伏预测精度", "没有时使用相似日和预测残差场景", "比较加入前后的滚动预测误差"],
  ["可选补充", "节假日、工作日和特殊事件", "问题二至四", "小区负荷可能因日历属性变化", "至少由日期构造星期和月份特征", "按工作日和休息日分组评估"],
  ["可选补充", "电池循环寿命和老化成本", "全部问题", "可限制无意义的频繁充放电", "设置小额等效成本并进行区间敏感性分析", "比较循环次数和费用"],
  ["可选补充", "电价发布与结算规则", "问题四", "决定未来电价是否在决策时已知", "分别给出已知价格下界和因果预测策略", "比较完全信息与预测策略差距"],
];
writeMatrix(supplements, 5, 0, supplementRows);
supplements.getRange("A6:F16").format.wrapText = true;
supplements.getRange("A6:F16").format.rowHeight = 58;
supplements.getRange("A:A").format.columnWidth = 14;
supplements.getRange("B:B").format.columnWidth = 34;
supplements.getRange("C:C").format.columnWidth = 15;
supplements.getRange("D:F").format.columnWidth = 45;
supplements.freezePanes.freezeRows(5);

// 异常检测图表
const chartsSheet = workbook.worksheets.add("异常检测图表");
titleBlock(chartsSheet, "异常值检测图表", "采用原始数据绑定的月度异常率和频数分布图；异常值均保留", "X");
const variables = ["小区负载功率", "光伏实际功率", "波动电价"];
const monthRows = [["月份", ...variables]];
for (let month = 1; month <= 12; month++) {
  monthRows.push([
    `${month}月`,
    ...variables.map((v) => monthlyAnomaly.find((r) => r["月份"] === month && r["变量"] === v)?.["异常率"] ?? 0),
  ]);
}
writeMatrix(chartsSheet, 3, 0, monthRows);
styleHeader(chartsSheet.getRange("A4:D4"));
chartsSheet.getRange("B5:D16").format.numberFormat = "0.00%";
chartsSheet.getRange("A:A").format.columnWidth = 10;
chartsSheet.getRange("B:D").format.columnWidth = 16;

const monthlyChart = chartsSheet.charts.add("line", chartsSheet.getRange("A4:D16"));
monthlyChart.title = "各月四分位距候选异常率";
monthlyChart.titleTextStyle.fontSize = 12;
monthlyChart.titleTextStyle.typeface = fontName;
monthlyChart.legend = { position: "top", textStyle: { typeface: fontName, fontSize: 10 } };
monthlyChart.xAxis = { axisType: "textAxis", textStyle: { typeface: fontName, fontSize: 9 } };
monthlyChart.yAxis = { numberFormatCode: "0.00%", numberFormatSourceLinked: false, textStyle: { typeface: fontName, fontSize: 9 } };
monthlyChart.setPosition("F4", "N18");

function addHistogram(variable, startColIndex, chartStart, chartEnd) {
  const rows = histograms
    .filter((r) => r["数据集"] === "全年实际数据" && r["变量"] === variable)
    .map((r) => [r["区间"], r["频数"]]);
  const headerRow = 20;
  const startLetter = colLetter(startColIndex);
  const nextLetter = colLetter(startColIndex + 1);
  writeMatrix(chartsSheet, headerRow, startColIndex, [["数值区间", "频数"], ...rows]);
  styleHeader(chartsSheet.getRange(`${startLetter}${headerRow + 1}:${nextLetter}${headerRow + 1}`));
  chartsSheet.getRange(`${startLetter}:${startLetter}`).format.columnWidth = 19;
  chartsSheet.getRange(`${nextLetter}:${nextLetter}`).format.columnWidth = 11;
  const lastRow = headerRow + rows.length + 1;
  const chart = chartsSheet.charts.add("bar", chartsSheet.getRange(`${startLetter}${headerRow + 1}:${nextLetter}${lastRow}`));
  chart.title = `${variable}频数分布`;
  chart.titleTextStyle.fontSize = 12;
  chart.titleTextStyle.typeface = fontName;
  chart.hasLegend = false;
  chart.xAxis = { axisType: "textAxis", textStyle: { typeface: fontName, fontSize: 8 } };
  chart.yAxis = { numberFormatCode: "#,##0", numberFormatSourceLinked: false, textStyle: { typeface: fontName, fontSize: 9 } };
  chart.setPosition(chartStart, chartEnd);
}

addHistogram("小区负载功率", 0, "A48", "H65");
addHistogram("光伏实际功率", 4, "I48", "P65");
addHistogram("波动电价", 8, "Q48", "X65");

// 最终统一基本样式
for (let i = 0; i < workbook.worksheets.items.length; i++) {
  const sheet = workbook.worksheets.getItemAt(i);
  const used = sheet.getUsedRange();
  if (used && !["全年实际处理后", "光伏预测处理后"].includes(sheet.name)) {
    used.format.verticalAlignment = "center";
  }
}

// 关键范围核验
const checks = [];
checks.push((await workbook.inspect({ kind: "table", range: "说明与汇总!A1:H36", include: "values,formulas", tableMaxRows: 40, tableMaxCols: 8, maxChars: 7000 })).ndjson);
checks.push((await workbook.inspect({ kind: "table", range: "异常值汇总!A1:P12", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 16, maxChars: 6000 })).ndjson);
checks.push((await workbook.inspect({ kind: "table", range: "附件1处理后!A1:J12", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 10, maxChars: 5000 })).ndjson);
checks.push((await workbook.inspect({ kind: "table", range: "全年实际处理后!A1:V10", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 22, maxChars: 6000 })).ndjson);
checks.push((await workbook.inspect({ kind: "table", range: "光伏预测处理后!A1:M10", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 13, maxChars: 5000 })).ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "最终公式错误扫描",
  maxChars: 6000,
});
console.log(checks.join("\n"));
console.log(errors.ndjson);

const previews = [
  ["说明与汇总", "A1:H36"],
  ["缺失值汇总", "A1:F18"],
  ["异常值汇总", "A1:P12"],
  ["异常值明细", "A1:I28"],
  ["附件1处理后", "A1:J24"],
  ["全年实际处理后", "A1:V22"],
  ["光伏预测处理后", "A1:M22"],
  ["标准化参数", "A1:H14"],
  ["数据补充建议", "A1:F16"],
  ["异常检测图表", "A1:X65"],
];
for (const [sheetName, range] of previews) {
  let preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safeName = sheetName.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(path.join(qaDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
  preview = null;
  if (global.gc) global.gc();
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`已生成：${outputPath}`);
